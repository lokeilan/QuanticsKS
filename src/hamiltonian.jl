# Build QTT operators from the model and density, then evaluate density and total energy.

build_operators(problem, accuracy, rec, physics, ::Val{:qtt}) =
    build_operators(problem, accuracy, rec, physics)

"Full-grid fields with sparse FD2 kinetics and FFT convolution."
struct DenseOperators{K,T,O}
    external::Vector{Float64}
    kinetic::SparseMatrixCSC{Float64,Int}
    interaction::K
    xc_table::T
    cartesian_order::O
    nuclear::Float64
end

struct DenseHamiltonian
    kinetic::SparseMatrixCSC{Float64,Int}
    potential::Vector{Float64}
end

function dense_kinetic(grid::PeriodicGrid1D, mass)
    n = length(grid)
    coefficient = 1 / (2mass * spacing(grid)^2)
    T = spdiagm(
        -1 => fill(-coefficient, n - 1),
        0 => fill(2coefficient, n),
        1 => fill(-coefficient, n - 1),
    )
    T[1, n] = T[n, 1] = -coefficient
    T
end

function dense_kinetic(grid::BoxGrid3D, mass)
    n = grid_shape(grid)
    rows, columns, values = Int[], Int[], Float64[]
    weights = 1 ./ (2mass .* spacing(grid) .^ 2)
    for j = 0:(length(grid) - 1)
        index = grid_indices(grid, j)
        push!(rows, j + 1)
        push!(columns, j + 1)
        push!(values, 2sum(weights))
        for a = 1:3, direction in (-1, 1)
            0 <= index[a] + direction < n[a] || continue
            neighbor = ntuple(b -> index[b] + (a == b ? direction : 0), 3)
            push!(rows, j + 1)
            push!(columns, grid_index(grid, neighbor) + 1)
            push!(values, -weights[a])
        end
    end
    sparse(rows, columns, values, length(grid), length(grid))
end

function build_operators(problem, accuracy, rec, physics, ::Val{:dense})
    validate(physics)
    grid, model = problem.grid, problem.model
    external = [
        external_value(model, grid_position(grid, j), grid isa PeriodicGrid1D ? grid.L : grid)
        for j = 0:(length(grid) - 1)
    ]
    interaction = if model isa SoftCoulombLSDA1D
        fft([kernel_value(model, grid_position(grid, j), grid.L) for j = 0:(length(grid) - 1)])
    elseif model isa AllElectronLDA3D
        n, h = grid_shape(grid), spacing(grid)
        kernel = map(CartesianIndices(2 .* n)) do index
            displacement = ntuple(a -> begin
                q = index[a] - 1
                (q < n[a] ? q : q - 2n[a]) * h[a]
            end, 3)
            coulomb_cell_average(displacement, h)
        end
        fft(kernel)
    else
        nothing
    end
    xc_table = if has_xc(model)
        Q.XCTable(
            npoints = physics.xc_table_points,
            density_min = physics.xc_density_min,
            density_max = physics.xc_density_max,
            rtol = physics.exchange_quadrature_rtol,
            parallel_tasks = get(rec.options, :parallel_tasks, 1),
        )
    elseif model isa AllElectronLDA3D && model.xc == :pz81
        PZ81Table()
    else
        nothing
    end
    order =
        grid isa BoxGrid3D ?
        map(index -> grid_index(grid, Tuple(index) .- 1) + 1, CartesianIndices(grid_shape(grid))) :
        nothing
    DenseOperators(
        external,
        dense_kinetic(grid, problem.mass),
        interaction,
        xc_table,
        order,
        nuclear_energy(model, grid isa PeriodicGrid1D ? grid.L : grid),
    )
end

function dense_hartree(rho, problem, ops::DenseOperators)
    ops.interaction === nothing && return zeros(length(rho))
    dV = volume_element(problem.grid)
    ops.cartesian_order === nothing && return dV .* real.(ifft(ops.interaction .* fft(rho)))
    n = grid_shape(problem.grid)
    padded = zeros(ComplexF64, 2 .* n)
    region = ntuple(a -> 1:n[a], 3)
    padded[region...] .= rho[ops.cartesian_order]
    potential = real.(ifft(ops.interaction .* fft(padded)))[region...]
    result = zeros(length(rho))
    result[vec(ops.cartesian_order)] .= dV .* vec(potential)
    result
end

function density(
    problem,
    orbitals::AbstractVector{<:AbstractVector{<:Number}},
    accuracy,
    rec;
    renormalize = false,
    label = "density",
)
    rho = zeros(length(problem.grid))
    for (psi, occupation) in zip(orbitals, problem.occupations)
        @. rho += occupation * abs2(psi)
    end
    rho ./= volume_element(problem.grid)
    if renormalize
        charge = volume_element(problem.grid) * sum(rho)
        charge > 0 || error("Nonpositive density integral")
        rho .*= sum(problem.occupations) / charge
    end
    rho
end

function effective_hamiltonian(
    problem,
    rho,
    ops::DenseOperators,
    accuracy,
    rec;
    fit_potential = true,
    time = 0.0,
    drive = NoDrive(),
)
    drive isa NoDrive || throw(ArgumentError("Dense orbitals currently support SCF only"))
    potential = ops.external + dense_hartree(rho, problem, ops)
    ops.xc_table === nothing || (potential .+= map(r -> Q.xc(ops.xc_table, r)[2], rho))
    DenseHamiltonian(ops.kinetic, potential)
end

green_operator(problem, accuracy, scf, physics, ops::DenseOperators) =
    cholesky(Hermitian(ops.kinetic + scf.green_shift * I))

function density_preconditioner(problem, ops::DenseOperators, accuracy, scf, rec)
    iszero(scf.dielectric_susceptibility) && return nothing
    problem.grid isa PeriodicGrid1D && problem.model isa SoftCoulombLSDA1D ||
        throw(ArgumentError("Dielectric density preconditioning requires periodic 1D soft Coulomb"))
    denominator =
        1 .+ scf.dielectric_susceptibility * volume_element(problem.grid) .* real.(ops.interaction)
    minimum(denominator) > 0 || error("Nonpositive dielectric eigenvalue")
    residual -> begin
        rhs = residual .- sum(residual) / length(residual)
        transformed = fft(rhs) ./ denominator
        transformed[1] = 0
        real.(ifft(transformed))
    end
end

function total_energy(
    problem,
    orbitals,
    rho,
    ops::DenseOperators,
    accuracy,
    rec;
    time = 0.0,
    drive = NoDrive(),
)
    drive isa NoDrive || throw(ArgumentError("Dense orbitals currently support SCF only"))
    dV = volume_element(problem.grid)
    kinetic = sum(
        occupation * real(dot(psi, ops.kinetic, psi)) for
        (psi, occupation) in zip(orbitals, problem.occupations)
    )
    external = dV * dot(rho, ops.external)
    hartree = dV / 2 * dot(rho, dense_hartree(rho, problem, ops))
    xc = ops.xc_table === nothing ? 0.0 : dV * sum(r -> Q.xc(ops.xc_table, r)[1], rho)
    nuclear = ops.nuclear
    (;
        total = kinetic + external + hartree + xc + nuclear,
        kinetic,
        external,
        hartree,
        xc,
        nuclear,
        drive = 0.0,
    )
end

Q.potential_samples(H::DenseHamiltonian, rec) = H.potential
Q.h_action(H::DenseHamiltonian, psi, cfg, rec; kwargs...) = H.kinetic * psi + H.potential .* psi
Q.h_residual(H::DenseHamiltonian, psi, energy, cfg, rec; kwargs...) =
    H.kinetic * psi + (H.potential .- energy) .* psi
Q.h_expectation(H::DenseHamiltonian, psi) = real(dot(psi, H.kinetic * psi + H.potential .* psi))
Q.h_residual_norm(H::DenseHamiltonian, psi, energy, cfg, rec; kwargs...) =
    norm(Q.h_residual(H, psi, energy, cfg, rec))
Q.h_project(::Nothing, psi, H::DenseHamiltonian; kwargs...) = Q.h_action(H, psi, nothing, nothing)
Q.h_project_tangent(::Nothing, psi, H::DenseHamiltonian) = Q.h_action(H, psi, nothing, nothing)
function Q.h_gram(orbitals, H::DenseHamiltonian; cfg = (;))
    X = reduce(hcat, orbitals)
    projected = X' * (H.kinetic * X + H.potential .* X)
    (projected + projected') / 2
end
Q.ritz_projection(orbitals, H::DenseHamiltonian, cfg, rec; kwargs...) = Q.h_gram(orbitals, H)
Q.green_rhs(H::DenseHamiltonian, psi, energy, cfg, rec; kwargs...) =
    (cfg.green_shift + energy .- H.potential) .* psi
function Q.block_green(orbitals, energies, H::DenseHamiltonian, green, cfg, rec; kwargs...)
    rhs = (cfg.green_shift .+ transpose(energies) .- H.potential) .* reduce(hcat, orbitals)
    updated = green \ rhs
    updated ./= sqrt.(sum(abs2, updated; dims = 1))
    collect.(eachcol(updated))
end

# Model-specific interactions supply the shared SCF and propagation workflows.
kernel_value(model::SoftCoulombLSDA1D, x, L) =
    model.coupling * Q.periodic_soft_coulomb(x / model.softening, L / model.softening) /
    model.softening
external_value(model::SoftCoulombLSDA1D, x, L) =
    -sum(
        charge * kernel_value(model, x - position, L) for
        (position, charge) in zip(model.positions, model.charges)
    )
function external_value(model::LocalPotential1D, x, L)
    value = model.potential(x)
    isreal(value) && isfinite(value) ||
        throw(ArgumentError("A finite real local potential is required"))
    real(value)
end
nuclear_energy(model::LocalPotential1D, L) = 0.0
nuclear_energy(model::SoftCoulombLSDA1D, L) = sum(
    (
        model.charges[i] *
        model.charges[j] *
        kernel_value(model, model.positions[i] - model.positions[j], L) for
        i in eachindex(model.positions) for j = 1:(i - 1)
    );
    init = 0.0,
)
has_xc(model) = model isa SoftCoulombLSDA1D && model.xc == :lsda

struct Operators{W,T,S}
    external::Q.QTT
    interaction::W
    xc_table::T
    plus::Vector{Array{ComplexF64,4}}
    minus::Vector{Array{ComplexF64,4}}
    external_samples::S
    nuclear::Float64
end
function build_operators(problem, accuracy, rec, physics::PhysicsNumerics)
    validate(physics)
    grid = problem.grid
    dx = spacing(grid)
    ngrid = length(grid)
    pivots =
        problem.model isa SoftCoulombLSDA1D ?
        round.(Int, mod.(problem.model.positions, grid.L) ./ dx) .% ngrid : Int[]
    model = deepcopy(problem.model)
    external_samples =
        Q.memoized_sample(Q.sampled(j -> external_value(model, j * dx, grid.L), grid.R), rec)
    built = Q.recorded_map(Any, 2, rec.options, rec) do item, localrec
        if item == 1
            problem.model isa LocalPotential1D && problem.model.potential isa ConstantPotential ?
            Q.constant_tt(grid.R, problem.model.potential.value) :
            Q.cross(
                localrec,
                grid.R,
                external_samples;
                tol = accuracy.external_tolerance,
                cap = field_caps(accuracy).external_cap,
                label = "external",
                pivots,
            )
        else
            problem.model isa SoftCoulombLSDA1D ?
            Q.circulant(
                Q.cross(
                    localrec,
                    grid.R,
                    j -> kernel_value(problem.model, j * dx, grid.L);
                    tol = accuracy.kernel_tolerance,
                    cap = field_caps(accuracy).kernel_cap,
                    label = "interaction",
                ),
            ) : nothing
        end
    end
    external, interaction = built
    xc_table =
        has_xc(problem.model) ?
        Q.XCTable(
            npoints = physics.xc_table_points,
            density_min = physics.xc_density_min,
            density_max = physics.xc_density_max,
            rtol = physics.exchange_quadrature_rtol,
            parallel_tasks = get(rec.options, :parallel_tasks, 1),
        ) : nothing
    Operators(
        external,
        interaction,
        xc_table,
        Q.shift_mpo(grid.R, 1),
        Q.shift_mpo(grid.R, -1),
        # All consumers reuse the static field fitted at its own accuracy budget.
        Q.memoized_sample(external, rec),
        nuclear_energy(model, grid.L),
    )
end

green_operator(p, a, s, physics, ops) =
    Q.green_mpo(p.grid.R, spacing(p.grid) * sqrt(p.mass), s.green_shift)

"Build the density update from the same discrete Hartree operator as the Hamiltonian."
function density_preconditioner(problem, ops, accuracy, scf, rec)
    iszero(scf.dielectric_susceptibility) && return nothing
    problem.grid isa PeriodicGrid1D && problem.model isa SoftCoulombLSDA1D ||
        throw(ArgumentError("Dielectric density preconditioning requires periodic 1D soft Coulomb"))
    cfg = solver_config(problem, accuracy, scf)
    residual -> Q.dielectric_update(residual, ops.interaction, cfg, rec)
end

# Cores store sqrt(dV)*psi, so divide by the volume element to recover density.
function density(problem, orbitals, accuracy, rec; renormalize = false, label = "density")
    dx = volume_element(problem.grid)
    orbital_samples = Q.block_evaluator(orbitals, rec)
    rho = Q.cross(
        rec,
        nsites(problem.grid),
        Q.DensitySamples(orbital_samples, Float64.(problem.occupations ./ dx));
        tol = accuracy.density_tolerance,
        cap = field_caps(accuracy).density_cap,
        label,
    )
    if renormalize
        charge = dx * real(sum(Q.tensortrain(rho)))
        charge > 0 || error("Nonpositive density integral")
        return Q.scale(rho, sum(problem.occupations) / charge)
    end
    rho
end
# Multiply the cyclic convolution by dx once to obtain the spatial integral.
hartree(rho, problem, ops, accuracy) =
    ops.interaction === nothing ? Q.constant_tt(problem.grid.R, 0) :
    Q.scale(
        Q.apply_mpo(
            ops.interaction,
            rho;
            tol = accuracy.algebra_tolerance,
            cap = field_caps(accuracy).hartree_cap,
        ),
        spacing(problem.grid),
    )
"Freeze one density and share its Hartree/XC expression across local consumers."
function local_potential_samples(problem, rho, ops, accuracy, rec; time = 0.0, drive = NoDrive())
    if problem.model isa LocalPotential1D && drive isa NoDrive
        return ops.external_samples
    end
    # Hartree and XC read the same immutable SCF input, never updated orbitals.
    frozen_density = Q.QTT([copy(core) for core in rho.c])
    hartree_potential = hartree(frozen_density, problem, ops, accuracy)
    rho_samples = Q.memoized_sample(frozen_density, rec)
    hartree_samples = Q.memoized_sample(hartree_potential, rec)
    dx = spacing(problem.grid)
    Q.memoized_sample(
        Q.pointwise(
            (external, hartree, density, drive) ->
                real(external + hartree) +
                (ops.xc_table === nothing ? 0.0 : Q.xc(ops.xc_table, real(density))[2]) +
                drive,
            ops.external_samples,
            hartree_samples,
            rho_samples,
            Q.sampled(j -> drive_value(drive, j * dx, time, problem.grid.L), problem.grid.R),
        ),
        rec,
    )
end

function effective_hamiltonian(
    problem,
    rho,
    ops,
    accuracy,
    rec;
    time = 0.0,
    drive = NoDrive(),
    fit_potential = true,
)
    field = local_potential_samples(problem, rho, ops, accuracy, rec; time, drive)
    # Core projections need one shared field representation; sampled residuals
    # and Green sources consume the original local expression directly.
    build_potential(fit_rec = rec) =
        problem.model isa LocalPotential1D &&
        drive isa NoDrive &&
        Q.rank(ops.external) <= field_caps(accuracy).effective_potential_cap ? ops.external :
        Q.cross(
            fit_rec,
            problem.grid.R,
            field;
            tol = accuracy.field_tolerance,
            cap = field_caps(accuracy).effective_potential_cap,
            label = "effective_potential",
        )
    potential = fit_potential ? build_potential() : nothing
    diagonal =
        fit_potential ? Q.diagonal(potential) :
        Q.DeferredPotentialMPO(localrec -> Q.diagonal(build_potential(something(localrec, rec))))
    Q.Hamiltonian(
        potential,
        diagonal,
        ops.plus,
        ops.minus,
        spacing(problem.grid) * sqrt(problem.mass),
        field,
    )
end
function total_energy(problem, orbitals, rho, ops, accuracy, rec; time = 0.0, drive = NoDrive())
    dx = spacing(problem.grid)
    # Contract FD2 energy directly, using exact gradients only for cancellation.
    kinetic = Q.kinetic_expectations(
        orbitals,
        ops.plus,
        ops.minus,
        dx * sqrt(problem.mass);
        cfg = rec.options,
    )
    kinetic_energy = sum(problem.occupations .* kinetic)
    external_energy = dx * real(Q.inner(rho, ops.external; cfg = rec.options))
    # Contract <rho|K|rho> directly instead of materializing the Hartree field.
    # V_H = dx * K*rho, so the Hartree integral carries dx^2.
    hartree_energy =
        ops.interaction === nothing ? 0.0 : 0.5dx^2 * real(Q.inner_mpo(rho, ops.interaction, rho))
    xc_energy = 0.0
    if ops.xc_table !== nothing
        rho_samples = Q.sampled(rho, rec)
        e = Q.cross(
            rec,
            problem.grid.R,
            Q.pointwise(value -> Q.xc(ops.xc_table, real(value))[1], rho_samples);
            tol = accuracy.xc_energy_tolerance,
            cap = field_caps(accuracy).xc_energy_cap,
            label = "xc_energy_density",
        )
        xc_energy = dx * real(sum(Q.tensortrain(e)))
    end
    drive_energy = 0.0
    if !(drive isa NoDrive)
        drive_potential = Q.cross(
            rec,
            problem.grid.R,
            j -> drive_value(drive, j * dx, time, problem.grid.L);
            tol = accuracy.field_tolerance,
            cap = field_caps(accuracy).drive_cap,
            label = "drive_energy",
        )
        drive_energy = dx * real(Q.inner(rho, drive_potential; cfg = rec.options))
    end
    ion_energy = ops.nuclear
    (;
        total = kinetic_energy +
                external_energy +
                hartree_energy +
                xc_energy +
                ion_energy +
                drive_energy,
        kinetic = kinetic_energy,
        external = external_energy,
        hartree = hartree_energy,
        xc = xc_energy,
        nuclear = ion_energy,
        drive = drive_energy,
    )
end

# Cartesian operators for isolated 3D systems.

"Lift a one-axis MPO into interleaved Cartesian sites with identity operators on the other axes."
function axis_mpo(w, axis::Int, axes = repeat(collect(1:3), length(w)))
    1 <= axis <= 3 || throw(ArgumentError("Axis must be 1:3"))
    count(==(axis), axes) == length(w) || throw(DimensionMismatch("Axis MPO sites"))
    out = Array{ComplexF64,4}[]
    bond = 1
    bit = 0
    for a in axes
        if a == axis
            bit += 1
            core = w[bit]
            push!(out, core)
            bond = size(core, 4)
        else
            core = zeros(ComplexF64, bond, 2, 2, bond)
            for b = 1:bond, s = 1:2
                core[b, s, s, b] = 1
            end
            push!(out, core)
        end
    end
    out
end

"Open 3D convolution with one signed displacement bit per axis."
cartesian_toeplitz(kernel::Q.QTT, R::Int) = cartesian_toeplitz(kernel, repeat(collect(1:3), R))
function cartesian_toeplitz(kernel::Q.QTT, axes::AbstractVector{Int})
    d = length(axes)
    length(kernel) == d + 3 || throw(DimensionMismatch("Signed kernel grid"))
    # The final three kernel bits encode the sign of each coordinate difference.
    cache = Q.TCI.TTCache(Q.tensortrain(kernel))
    tail = hcat([Q.TCI.evaluateright(cache, Q.bits(mask, 3)) for mask = 0:7]...)
    out = Array{ComplexF64,4}[]
    for k = 1:d
        g = kernel.c[k]
        rl, _, rr = size(g)
        axis = axes[k]
        w = zeros(ComplexF64, k == 1 ? 1 : 8rl, 2, 2, k == d ? 1 : 8rr)
        # One borrow bit per axis implements signed displacement without periodic wrap.
        for mask in (k == 1 ? (0:0) : (0:7)), i = 0:1, j = 0:1
            z = i - j - ((mask >> (axis - 1)) & 1)
            next = (mask & ~(1 << (axis - 1))) | ((z < 0 ? 1 : 0) << (axis - 1))
            slice = g[:, mod(z, 2) + 1, :]
            if k == d
                w[(1:rl) .+ mask * rl, i + 1, j + 1, 1] = slice * tail[:, next + 1]
            else
                w[(1:rl) .+ mask * rl, i + 1, j + 1, (1:rr) .+ next * rr] = slice
            end
        end
        push!(out, w)
    end
    out
end

# Integrate 1/r over a source voxel, including its finite self interaction.
function coulomb_primitive(x, y, z)
    r = sqrt(x * x + y * y + z * z)
    r == 0 && return 0.0
    logterm(a, b, c) = a * b == 0 ? 0.0 : a * b * log(c + r)
    atanterm(a, b, c) = a == 0 ? 0.0 : -a * a / 2 * atan(b * c / (a * r))
    logterm(x, y, z) +
    logterm(y, z, x) +
    logterm(z, x, y) +
    atanterm(x, y, z) +
    atanterm(y, z, x) +
    atanterm(z, x, y)
end
function coulomb_cell_average(r, h)
    # Gaussian quadrature avoids cancellation of large corner primitives far away.
    if sqrt(sum(abs2, r)) > 4maximum(h)
        nodes = (-sqrt(3 / 5), 0.0, sqrt(3 / 5))
        weights = (5 / 18, 4 / 9, 5 / 18)
        return sum(
            weights[i] * weights[j] * weights[k] /
            sqrt(sum((r[a] + h[a] * nodes[(i, j, k)[a]] / 2)^2 for a = 1:3)) for i = 1:3, j = 1:3,
            k = 1:3
        )
    end
    sum(
        prod(s) * coulomb_primitive((r .+ h .* s ./ 2)...) for
        s in Iterators.product((-1, 1), (-1, 1), (-1, 1))
    ) / prod(h)
end

"PZ81 unpolarized energy density and its density derivative, in atomic units."
function lda_pz81(rho::Real)
    isfinite(rho) || throw(ArgumentError("Nonfinite electron density"))
    rho <= 0 && return (0.0, 0.0)
    # rs is the Wigner-Seitz radius; ex/ec are energies per electron.
    # Perdew-Zunger correlation: https://doi.org/10.1103/PhysRevB.23.5048
    rs = cbrt(3 / (4pi * rho))
    ex = -0.75cbrt(3rho / pi)
    ec, derivative = if rs < 1
        A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
        (A * log(rs) + B + C * rs * log(rs) + D * rs, A / rs + C * (log(rs) + 1) + D)
    else
        gamma, b1, b2 = -0.1423, 1.0529, 0.3334
        den = 1 + b1 * sqrt(rs) + b2 * rs
        (gamma / den, -gamma * (b1 / (2sqrt(rs)) + b2) / den^2)
    end
    (rho * (ex + ec), 4ex / 3 + ec - rs * derivative / 3)
end
struct PZ81Table end
Q.xc(::PZ81Table, rho) = lda_pz81(rho)

struct CartesianOperators{W,T,S}
    external::Q.QTT
    interaction::W
    xc_table::T
    plus::NTuple{3,Vector{Array{ComplexF64,4}}}
    minus::NTuple{3,Vector{Array{ComplexF64,4}}}
    external_samples::S
    nuclear::Float64
end
function external_value(m::AllElectronLDA3D, r, g::BoxGrid3D)
    -sum(z * coulomb_cell_average(r .- p, spacing(g)) for (p, z) in zip(m.positions, m.charges))
end
function external_value(m::LocalPotential3D, r, g::BoxGrid3D)
    v = m.potential(r...)
    isreal(v) && isfinite(v) || throw(ArgumentError("A finite real potential is required"))
    real(v)
end
nuclear_energy(m::LocalPotential3D, g::BoxGrid3D) = 0.0
nuclear_energy(m::AllElectronLDA3D, g::BoxGrid3D) = sum(
    (
        m.charges[i] * m.charges[j] / sqrt(sum(abs2, m.positions[i] .- m.positions[j])) for
        i in eachindex(m.positions) for j = 1:(i - 1)
    );
    init = 0.0,
)

function build_operators(
    problem::Problem{<:AbstractModel3D},
    accuracy,
    rec,
    physics::PhysicsNumerics,
)
    validate(physics)
    grid = problem.grid
    model = deepcopy(problem.model)
    samples = Q.memoized_sample(
        Q.sampled(j -> external_value(model, grid_position(grid, j), grid), nsites(grid)),
        rec,
    )
    pivots =
        model isa AllElectronLDA3D ?
        [
            grid_index(
                grid,
                ntuple(
                    a -> clamp(
                        floor(Int, (r[a] + grid.L[a] / 2) / spacing(grid)[a]),
                        0,
                        grid_shape(grid)[a] - 1,
                    ),
                    3,
                ),
            ) for r in model.positions
        ] : Int[]
    external = Q.cross(
        rec,
        nsites(grid),
        samples;
        tol = accuracy.external_tolerance,
        cap = field_caps(accuracy).external_cap,
        label = "external",
        pivots,
    )
    interaction = if model isa AllElectronLDA3D
        n = grid_shape(grid)
        kernel = Q.cross(
            rec,
            nsites(grid) + 3,
            j -> begin
                q = grid_indices(grid, j & (length(grid) - 1))
                signs = j >> nsites(grid)
                r = ntuple(a -> (q[a] - ((signs >> (a - 1)) & 1) * n[a]) * spacing(grid)[a], 3)
                coulomb_cell_average(r, spacing(grid))
            end;
            tol = accuracy.kernel_tolerance,
            cap = field_caps(accuracy).kernel_cap,
            label = "interaction",
            pivots = [0],
        )
        cartesian_toeplitz(kernel, grid_axes(grid))
    else
        nothing
    end
    CartesianOperators(
        external,
        interaction,
        model isa AllElectronLDA3D && model.xc == :pz81 ? PZ81Table() : nothing,
        ntuple(
            axis -> axis_mpo(
                Q.shift_mpo(axis_bits(grid)[axis], 1; boundary = :open),
                axis,
                grid_axes(grid),
            ),
            3,
        ),
        ntuple(
            axis -> axis_mpo(
                Q.shift_mpo(axis_bits(grid)[axis], -1; boundary = :open),
                axis,
                grid_axes(grid),
            ),
            3,
        ),
        samples,
        nuclear_energy(model, grid),
    )
end

# Cartesian fields, Hamiltonian actions, and Green solves.

struct MPOContractionSamples <: Q.SampledFunction
    contraction::Q.TCI.Contraction{ComplexF64}
    R::Int
    cache_bytes::Int
end
Q.sample_length(f::MPOContractionSamples) = f.R
Q.uses_batch_contraction(::MPOContractionSamples) = true
Q.uses_tt_cache(::MPOContractionSamples) = true
Q.sampling_method(::MPOContractionSamples) = "mpo_contraction"
function trim_contraction_cache!(f::MPOContractionSamples)
    caches = (f.contraction.leftcache, f.contraction.rightcache)
    if sum(sum(sizeof, values(c); init = 0) for c in caches) > f.cache_bytes
        foreach(empty!, caches)
    end
    nothing
end
function (f::MPOContractionSamples)(j::Int)
    value = Q.TCI.evaluate(f.contraction, [(s, 1) for s in Q.bits(j, f.R)])
    trim_contraction_cache!(f)
    value
end
function (f::MPOContractionSamples)(left, right, middle::Val)
    value = Q.TCI.batchevaluate(f.contraction, left, right, middle)
    trim_contraction_cache!(f)
    value
end

function hartree(
    rho,
    problem::Problem{<:AbstractModel3D},
    ops,
    accuracy,
    rec = recorder(nothing, TCIOptions()),
)
    ops.interaction === nothing && return Q.constant_tt(nsites(problem.grid), 0)
    # Contract complete samples before truncating; a partial displacement-MPO contraction is not isometric.
    column = Q.TCI.TensorTrain([reshape(c, size(c, 1), 2, 1, size(c, 3)) for c in rho.c])
    samples = MPOContractionSamples(
        Q.TCI.Contraction(Q.TCI.TensorTrain(ops.interaction), column),
        nsites(problem.grid),
        rec.options.orbital_cache_max_bytes,
    )
    Q.scale(
        Q.cross(
            rec,
            nsites(problem.grid),
            samples;
            tol = accuracy.algebra_tolerance,
            cap = field_caps(accuracy).hartree_cap,
            label = "hartree",
        ),
        volume_element(problem.grid),
    )
end
function local_potential_samples(
    problem::Problem{<:AbstractModel3D},
    rho,
    ops,
    accuracy,
    rec;
    time = 0.0,
    drive = NoDrive(),
)
    drive isa NoDrive || throw(ArgumentError("3D time-dependent drives are not implemented"))
    problem.model isa LocalPotential3D && return ops.external_samples
    vh = Q.memoized_sample(hartree(rho, problem, ops, accuracy, rec), rec)
    Q.memoized_sample(
        Q.pointwise(
            (v, h, n) -> real(v + h) + (ops.xc_table === nothing ? 0.0 : lda_pz81(real(n))[2]),
            ops.external_samples,
            vh,
            Q.memoized_sample(rho, rec),
        ),
        rec,
    )
end

struct CartesianHamiltonian{S,D}
    potential::Union{Nothing,Q.QTT}
    diagonal::D
    plus::NTuple{3,Vector{Array{ComplexF64,4}}}
    minus::NTuple{3,Vector{Array{ComplexF64,4}}}
    grid::BoxGrid3D
    mass::Float64
    samples::S
end
Q.potential_samples(H::CartesianHamiltonian, rec) = H.samples
function effective_hamiltonian(
    problem::Problem{<:AbstractModel3D},
    rho,
    ops::CartesianOperators,
    accuracy,
    rec;
    fit_potential = true,
    kwargs...,
)
    field = local_potential_samples(problem, rho, ops, accuracy, rec; kwargs...)
    build_potential(fit_rec = rec) =
        problem.model isa LocalPotential3D &&
        Q.rank(ops.external) <= field_caps(accuracy).effective_potential_cap ? ops.external :
        Q.cross(
            fit_rec,
            nsites(problem.grid),
            field;
            tol = accuracy.field_tolerance,
            cap = field_caps(accuracy).effective_potential_cap,
            label = "effective_potential",
        )
    potential = fit_potential ? build_potential() : nothing
    diagonal =
        fit_potential ? Q.diagonal(potential) :
        Q.DeferredPotentialMPO(localrec -> Q.diagonal(build_potential(something(localrec, rec))))
    CartesianHamiltonian(
        potential,
        diagonal,
        ops.plus,
        ops.minus,
        problem.grid,
        problem.mass,
        field,
    )
end

"Keep all FD2 terms until the caller combines and truncates its final expression."
function kinetic_expression(H::CartesianHamiltonian, psi)
    weights = Float64[sum(1 ./ (H.mass .* spacing(H.grid) .^ 2))]
    terms = Q.QTT[psi]
    for a = 1:3, w in (H.plus[a], H.minus[a])
        push!(terms, Q.apply_mpo(w, psi; tol = 0.0, cap = 2Q.rank(psi)))
        push!(weights, -1 / (2H.mass * spacing(H.grid)[a]^2))
    end
    Q.linear_expression(terms, weights)
end
function Q.h_core_fallback(H::CartesianHamiltonian, psi, energy; tol, cap, rec = nothing)
    kinetic = kinetic_expression(H, psi)
    potential = Q.apply_mpo(Q.potential_mpo(H, rec), psi; tol, cap)
    Q.reconstruct_expression(
        Q.linear_expression((kinetic, potential, psi), (1, 1, -energy));
        tol,
        cap,
    )
end
function hamiltonian_samples(H::CartesianHamiltonian, ev, v, R; energy = 0.0)
    n, h = grid_shape(H.grid), spacing(H.grid)
    fn =
        j -> begin
            indices = grid_indices(H.grid, j)
            center = ev(j)
            kinetic = 0.0im
            for a = 1:3
                left =
                    indices[a] == 0 ? 0.0 :
                    ev(grid_index(H.grid, ntuple(b -> indices[b] - (a == b), 3)))
                right =
                    indices[a] == n[a] - 1 ? 0.0 :
                    ev(grid_index(H.grid, ntuple(b -> indices[b] + (a == b), 3)))
                kinetic += ((center - left) + (center - right)) / (2H.mass * h[a]^2)
            end
            kinetic + (v(j) - energy) * center
        end
    Q.sampled(fn, R)
end
function Q.h_residual(
    H::CartesianHamiltonian,
    psi,
    energy,
    cfg,
    rec;
    tol = cfg.algebra_tolerance,
    cap = cfg.work_cap,
    label = "orbital_residual",
    potential = nothing,
)
    Q.validate_truncation(tol, cap)
    isfinite(energy) || throw(ArgumentError("Nonfinite orbital energy"))
    fn = hamiltonian_samples(
        H,
        Q.evaluate(psi, rec),
        potential === nothing ? H.samples : potential,
        nsites(H.grid);
        energy,
    )
    Q.cross(
        rec,
        nsites(H.grid),
        fn;
        tol,
        cap,
        label,
        initial_zero_fallback = () -> Q.h_core_fallback(H, psi, energy; tol, cap, rec),
    )
end
function Q.h_project(tangent_basis, psi, H::CartesianHamiltonian; projected_psi = nothing)
    result = Q.project(tangent_basis, psi, Q.potential_mpo(H))
    center = projected_psi === nothing ? Q.project(tangent_basis, psi) : projected_psi
    for a = 1:3
        result += Q.kinetic_project(
            tangent_basis,
            psi,
            H.plus[a],
            H.minus[a],
            sqrt(H.mass) * spacing(H.grid)[a],
            center,
        )
    end
    result
end
Q.h_project_tangent(tangent_basis, z, H::CartesianHamiltonian) =
    Q.h_project(tangent_basis, Q.tangent_vector(tangent_basis, z), H)

function Q.h_gram(orbitals, H::CartesianHamiltonian; cfg = (;))
    Q.with_sampling_seed(0) do
        bras, kets, operators = Q.mpo_basis(orbitals, (Q.potential_mpo(H), H.plus..., H.minus...))
        potential, plus, minus = operators[1], operators[2:4], operators[5:7]
        # Build gradients only when direct FD2 contractions lose significant digits.
        gradients = Dict{Tuple{Int,Int},Q.QTT}()
        gradient(i, a) =
            get!(gradients, (i, a)) do
                Q.shift_difference(H.plus[a], orbitals[i])
            end
        faces = Dict{Int,Vector{Array{ComplexF64,4}}}()
        face(a) =
            get!(faces, a) do
                axis_mpo(
                    Q.diagonal(Q.delta_tt(axis_bits(H.grid)[a], grid_shape(H.grid)[a] - 1)),
                    a,
                    grid_axes(H.grid),
                )
            end
        out = zeros(ComplexF64, length(orbitals), length(orbitals))
        for j in eachindex(orbitals), i = 1:j
            value = Q.ITensorMPS.inner(bras[i], potential, kets[j]; make_inds_match = false)
            overlap = Q.ITensorMPS.inner(kets[i], kets[j]; make_inds_match = false)
            for a = 1:3
                terms = (
                    overlap,
                    -Q.ITensorMPS.inner(bras[i], plus[a], kets[j]; make_inds_match = false) / 2,
                    -Q.ITensorMPS.inner(bras[i], minus[a], kets[j]; make_inds_match = false) / 2,
                )
                numerator = Q.compensated_sum(terms)
                if abs(numerator) <= sqrt(eps(Float64)) * sum(abs, terms) ||
                   (i == j && real(numerator) < 0)
                    # An open shift loses the outer face; include its positive contribution.
                    numerator =
                        (
                            Q.inner(gradient(i, a), gradient(j, a)) +
                            Q.inner_mpo(orbitals[i], face(a), orbitals[j])
                        ) / 2
                end
                value += numerator / (H.mass * spacing(H.grid)[a]^2)
            end
            out[i, j] = value
            out[j, i] = conj(value)
            i == j && (out[i, i] = real(value))
        end
        out
    end
end

"QTT CG inverse of the discrete shifted kinetic operator; never a continuum surrogate."
struct CartesianGreen
    kinetic::CartesianHamiltonian
    shift::Float64
    tolerance::Float64
    max_iterations::Int
    cap::Int
end
function green_operator(
    problem::Problem{<:AbstractModel3D},
    accuracy,
    scf,
    physics,
    ops::CartesianOperators,
)
    zero = Q.constant_tt(nsites(problem.grid), 0)
    H = CartesianHamiltonian(
        zero,
        Q.diagonal(zero),
        ops.plus,
        ops.minus,
        problem.grid,
        problem.mass,
        nothing,
    )
    CartesianGreen(
        H,
        scf.green_shift,
        physics.green_relative_tolerance,
        physics.green_max_iterations,
        accuracy.work_cap,
    )
end

"Residual of the stored state under the untruncated discrete Green operator."
function green_residual(G::CartesianGreen, psi, rhs)
    Q.linear_expression((rhs, kinetic_expression(G.kinetic, psi), psi), (1, -1, -G.shift))
end

function Q.apply_mpo(
    G::CartesianGreen,
    rhs::Q.QTT;
    tol = 1e-11,
    cap = G.cap,
    rec = recorder(nothing, TCIOptions()),
)
    Q.validate_truncation(tol, cap)
    target = G.tolerance
    work_tolerance = min(max(tol, eps(Float64)), target / 100)
    work_cap = max(cap, G.cap)
    combine(xs, ws) = Q.linear_combination(
        rec,
        xs,
        ws;
        tol = work_tolerance,
        cap = work_cap,
        label = "green_combination",
    )
    # Evaluate all six neighbors in one FD2 expression, without constructing shifted states.
    accuracy = (; algebra_tolerance = work_tolerance, work_cap)
    action(x) = Q.h_residual(
        G.kinetic,
        x,
        -G.shift,
        accuracy,
        rec;
        potential = _ -> 0.0,
        label = "green_action",
    )
    rhs_norm = Q.ttnorm(rhs)
    rhs_norm == 0 && return Q.constant_tt(length(rhs), 0)
    x, r, direction = Q.constant_tt(length(rhs), 0), rhs, rhs
    residual_norm_squared = real(Q.inner(r, r))
    relative_residual = 1.0
    for iteration = 1:(G.max_iterations)
        applied_direction = action(direction)
        curvature = real(Q.inner(direction, applied_direction))
        curvature > 0 || error("3D Green CG lost positive curvature; increase work_cap")
        alpha = residual_norm_squared / curvature
        x = combine((x, direction), (1, alpha))
        residual = combine((r, applied_direction), (1, -alpha))
        next_residual_norm_squared = real(Q.inner(residual, residual))
        relative_residual = sqrt(max(next_residual_norm_squared, 0)) / rhs_norm
        if sqrt(max(next_residual_norm_squared, 0)) <= target * rhs_norm
            result = Q.rounded(x; tol, cap)
            expression = green_residual(G, result, rhs)
            relative_residual = sqrt(real(only(Q.expression_gram([expression])))) / rhs_norm
            relative_residual <= target && return result
            # Truncation is allowed for the next search direction, not for acceptance.
            true_residual =
                Q.reconstruct_expression(expression; tol = work_tolerance, cap = work_cap)
            x, r, direction = result, true_residual, true_residual
            residual_norm_squared = real(Q.inner(r, r))
        else
            direction = combine(
                (residual, direction),
                (1, next_residual_norm_squared / residual_norm_squared),
            )
            r, residual_norm_squared = residual, next_residual_norm_squared
        end
    end
    result = Q.rounded(x; tol, cap)
    expression = green_residual(G, result, rhs)
    relative_residual = sqrt(real(only(Q.expression_gram([expression])))) / rhs_norm
    relative_residual <= target && return result
    error(
        "3D Green CG residual $relative_residual exceeds tolerance $(G.tolerance) after $(G.max_iterations) iterations; work cap=$work_cap",
    )
end
Q.project(f::Q.TangentSpace, psi::Q.QTT, G::CartesianGreen) =
    Q.project(f, Q.apply_mpo(G, psi; tol = G.tolerance / 100, cap = G.cap))

function total_energy(
    problem::Problem{<:AbstractModel3D},
    orbitals,
    rho,
    ops::CartesianOperators,
    accuracy,
    rec;
    time = 0.0,
    drive = NoDrive(),
)
    drive isa NoDrive || throw(ArgumentError("3D drives are not implemented"))
    zero = Q.constant_tt(nsites(problem.grid), 0)
    kinetic_operator = CartesianHamiltonian(
        zero,
        Q.diagonal(zero),
        ops.plus,
        ops.minus,
        problem.grid,
        problem.mass,
        nothing,
    )
    kinetic = sum(
        Q.recorded_map(Float64, length(orbitals), rec.options, rec; random = false) do i, localrec
            problem.occupations[i] * Q.h_expectation(kinetic_operator, orbitals[i])
        end,
    )
    dv = volume_element(problem.grid)
    external = dv * real(Q.inner(rho, ops.external))
    # V_H = dV * K*rho, so the Hartree integral carries dV^2.
    hartree_energy =
        ops.interaction === nothing ? 0.0 : dv^2 / 2 * real(Q.inner_mpo(rho, ops.interaction, rho))
    xc_energy = if ops.xc_table === nothing
        0.0
    else
        xc_energy_density = Q.cross(
            rec,
            nsites(problem.grid),
            Q.pointwise(n -> lda_pz81(real(n))[1], Q.sampled(rho, rec));
            tol = accuracy.xc_energy_tolerance,
            cap = field_caps(accuracy).xc_energy_cap,
            label = "xc_energy_density",
        )
        dv * real(sum(Q.tensortrain(xc_energy_density)))
    end
    (;
        total = kinetic + external + hartree_energy + xc_energy + ops.nuclear,
        kinetic,
        external,
        hartree = hartree_energy,
        xc = xc_energy,
        nuclear = ops.nuclear,
        drive = 0.0,
    )
end
