# Build QTT operators from the model and density, then evaluate density and total energy.

# Model-specific interactions supply the shared SCF and propagation workflows.
kernel_value(model::SoftCoulombLSDA1D, x, L) =
    model.coupling * Q.soft(x / model.softening, L / model.softening) / model.softening
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
        external_samples,
        nuclear_energy(model, grid.L),
    )
end

green_operator(p, a, s, physics, ops) =
    Q.green_mpo(p.grid.R, spacing(p.grid) * sqrt(p.mass), s.green_shift)

# Cores store sqrt(dV)*psi, so divide by the volume element to recover density.
function density(problem, orbitals, accuracy, rec; renormalize = false, label = "density")
    dx = volume_element(problem.grid)
    orbital_samples = Q.block_oracle(orbitals, rec)
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
    Vhartree = hartree(frozen_density, problem, ops, accuracy)
    rho_samples = Q.memoized_sample(frozen_density, rec)
    hartree_samples = Q.memoized_sample(Vhartree, rec)
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

function effective_hamiltonian(problem, rho, ops, accuracy, rec; time = 0.0, drive = NoDrive())
    field = local_potential_samples(problem, rho, ops, accuracy, rec; time, drive)
    # Core projections need one shared field representation; sampled residuals
    # and Green sources consume the original local expression directly.
    Veff =
        problem.model isa LocalPotential1D &&
        drive isa NoDrive &&
        Q.rank(ops.external) <= field_caps(accuracy).effective_potential_cap ? ops.external :
        Q.cross(
            rec,
            problem.grid.R,
            field;
            tol = accuracy.field_tolerance,
            cap = field_caps(accuracy).effective_potential_cap,
            label = "effective_potential",
        )
    Q.Hamiltonian(
        Veff,
        Q.diagonal(Veff),
        ops.plus,
        ops.minus,
        spacing(problem.grid) * sqrt(problem.mass),
        field,
    )
end
function total_energy(problem, orbitals, rho, ops, accuracy, rec; time = 0.0, drive = NoDrive())
    dx = spacing(problem.grid)
    kinetic_terms =
        Q.recorded_map(Float64, length(orbitals), rec.options, rec; random = false) do i, localrec
            occupation, psi = problem.occupations[i], orbitals[i]
            shifted = Q.apply_mpo(
                ops.plus,
                psi;
                tol = accuracy.algebra_tolerance,
                cap = accuracy.orbital_cap,
            )
            gradient_norm =
                Q.difference_norm(shifted, psi; rec = localrec, label = "kinetic_difference_norm")
            occupation * gradient_norm^2 / (2problem.mass * dx^2)
        end
    kinetic_energy = sum(kinetic_terms)
    Vhartree = hartree(rho, problem, ops, accuracy)
    external_energy = dx * real(Q.inner(rho, ops.external; cfg = rec.options))
    hartree_energy = 0.5dx * real(Q.inner(rho, Vhartree; cfg = rec.options))
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
        Veff = Q.cross(
            rec,
            problem.grid.R,
            j -> drive_value(drive, j * dx, time, problem.grid.L);
            tol = accuracy.field_tolerance,
            cap = field_caps(accuracy).drive_cap,
            label = "drive_energy",
        )
        drive_energy = dx * real(Q.inner(rho, Veff; cfg = rec.options))
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

"Lift a one-axis MPO into interleaved Cartesian sites with identity spectators."
function axis_mpo(w, axis::Int)
    1 <= axis <= 3 || throw(ArgumentError("Axis must be 1:3"))
    out = Array{ComplexF64,4}[]
    bond = 1
    for k = 1:(3length(w))
        if mod1(k, 3) == axis
            core = w[cld(k, 3)]
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

"Open 3D convolution from a signed displacement QTT on (2N)^3 points."
function cartesian_toeplitz(kernel::Q.QTT, R::Int)
    length(kernel) == 3(R + 1) || throw(DimensionMismatch("Signed kernel grid"))
    d = 3R
    # The final three kernel bits encode the sign of each coordinate difference.
    cache = Q.TCI.TTCache(Q.tensortrain(kernel))
    tail = hcat([Q.TCI.evaluateright(cache, Q.bits(mask, 3)) for mask = 0:7]...)
    out = Array{ComplexF64,4}[]
    for k = 1:d
        g = kernel.c[k]
        rl, _, rr = size(g)
        axis = mod1(k, 3)
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

function build_operators(p::Problem{<:AbstractModel3D}, a, rec, physics::PhysicsNumerics)
    validate(physics)
    g = p.grid
    m = deepcopy(p.model)
    samples =
        Q.memoized_sample(Q.sampled(j -> external_value(m, grid_position(g, j), g), nsites(g)), rec)
    pivots =
        m isa AllElectronLDA3D ?
        [
            grid_index(
                g,
                ntuple(
                    a -> clamp(floor(Int, (r[a] + g.L[a] / 2) / spacing(g)[a]), 0, (1 << g.R) - 1),
                    3,
                ),
            ) for r in m.positions
        ] : Int[]
    ext = Q.cross(
        rec,
        nsites(g),
        samples;
        tol = a.external_tolerance,
        cap = field_caps(a).external_cap,
        label = "external",
        pivots,
    )
    interaction = if m isa AllElectronLDA3D
        n = 1 << g.R
        kernel = Q.cross(
            rec,
            nsites(g) + 3,
            j -> begin
                q = ntuple(a -> sum(((j >> (3b + a - 1)) & 1) << b for b = 0:(g.R)), 3)
                r = ntuple(a -> (q[a] < n ? q[a] : q[a] - 2n) * spacing(g)[a], 3)
                coulomb_cell_average(r, spacing(g))
            end;
            tol = a.kernel_tolerance,
            cap = field_caps(a).kernel_cap,
            label = "interaction",
            pivots = [0],
        )
        cartesian_toeplitz(kernel, g.R)
    else
        nothing
    end
    CartesianOperators(
        ext,
        interaction,
        m isa AllElectronLDA3D && m.xc == :pz81 ? PZ81Table() : nothing,
        ntuple(axis -> axis_mpo(Q.shift_mpo(g.R, 1; boundary = :open), axis), 3),
        ntuple(axis -> axis_mpo(Q.shift_mpo(g.R, -1; boundary = :open), axis), 3),
        samples,
        nuclear_energy(m, g),
    )
end

# Cartesian fields, Hamiltonian actions, and Green solves.

struct MPOContractionSamples <: Q.SampledFunction
    contraction::Q.TCI.Contraction{ComplexF64}
    R::Int
    cache_bytes::Int
end
Q.sample_length(f::MPOContractionSamples) = f.R
Q.native_batch(::MPOContractionSamples) = true
Q.uses_tt_frames(::MPOContractionSamples) = true
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

function hartree(rho, p::Problem{<:AbstractModel3D}, ops, a, rec = recorder(nothing, TCIOptions()))
    ops.interaction === nothing && return Q.constant_tt(nsites(p.grid), 0)
    # Contract complete samples before truncating; a raw borrow-MPO prefix is not isometric.
    column = Q.TCI.TensorTrain([reshape(c, size(c, 1), 2, 1, size(c, 3)) for c in rho.c])
    samples = MPOContractionSamples(
        Q.TCI.Contraction(Q.TCI.TensorTrain(ops.interaction), column),
        nsites(p.grid),
        rec.options.orbital_cache_max_bytes,
    )
    Q.scale(
        Q.cross(
            rec,
            nsites(p.grid),
            samples;
            tol = a.algebra_tolerance,
            cap = field_caps(a).hartree_cap,
            label = "hartree",
        ),
        volume_element(p.grid),
    )
end
function local_potential_samples(
    p::Problem{<:AbstractModel3D},
    rho,
    ops,
    a,
    rec;
    time = 0.0,
    drive = NoDrive(),
)
    drive isa NoDrive || throw(ArgumentError("3D time-dependent drives are not implemented"))
    p.model isa LocalPotential3D && return ops.external_samples
    vh = Q.memoized_sample(hartree(rho, p, ops, a, rec), rec)
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

struct CartesianHamiltonian{S}
    potential::Q.QTT
    diagonal::Vector{Array{ComplexF64,4}}
    plus::NTuple{3,Vector{Array{ComplexF64,4}}}
    minus::NTuple{3,Vector{Array{ComplexF64,4}}}
    grid::BoxGrid3D
    mass::Float64
    samples::S
end
Q.potential_samples(H::CartesianHamiltonian, rec) = H.samples
function effective_hamiltonian(p::Problem{<:AbstractModel3D}, rho, ops, a, rec; kwargs...)
    field = local_potential_samples(p, rho, ops, a, rec; kwargs...)
    v =
        p.model isa LocalPotential3D &&
        Q.rank(ops.external) <= field_caps(a).effective_potential_cap ? ops.external :
        Q.cross(
            rec,
            nsites(p.grid),
            field;
            tol = a.field_tolerance,
            cap = field_caps(a).effective_potential_cap,
            label = "effective_potential",
        )
    CartesianHamiltonian(v, Q.diagonal(v), ops.plus, ops.minus, p.grid, p.mass, field)
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
function Q.h_core_fallback(H::CartesianHamiltonian, psi, energy; tol, cap)
    kinetic = kinetic_expression(H, psi)
    potential = Q.apply_mpo(H.diagonal, psi; tol, cap)
    Q.materialize_expression(
        Q.linear_expression((kinetic, potential, psi), (1, 1, -energy));
        tol,
        cap,
    )
end
function Q.h_residual(
    H::CartesianHamiltonian,
    psi,
    energy,
    cfg,
    rec;
    tol = cfg.round_tol,
    cap = cfg.workcap,
    label = "orbital_residual",
    potential = nothing,
)
    Q.validate_truncation(tol, cap)
    isfinite(energy) || throw(ArgumentError("Nonfinite orbital energy"))
    ev = Q.evaluate(psi, rec)
    v = potential === nothing ? H.samples : potential
    n, h = 1 << H.grid.R, spacing(H.grid)
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
                    indices[a] == n - 1 ? 0.0 :
                    ev(grid_index(H.grid, ntuple(b -> indices[b] + (a == b), 3)))
                kinetic += ((center - left) + (center - right)) / (2H.mass * h[a]^2)
            end
            kinetic + (v(j) - energy) * center
        end
    Q.cross(
        rec,
        nsites(H.grid),
        fn;
        tol,
        cap,
        label,
        initial_zero_fallback = () -> Q.h_core_fallback(H, psi, energy; tol, cap),
    )
end
function Q.h_project(f, psi, H::CartesianHamiltonian)
    result = Q.project(f, psi, H.diagonal)
    center = Q.project(f, psi)
    for a = 1:3
        result +=
            (center - (Q.project(f, psi, H.plus[a]) + Q.project(f, psi, H.minus[a])) / 2) /
            (H.mass * spacing(H.grid)[a]^2)
    end
    result
end
Q.h_project_tangent(f, z, H::CartesianHamiltonian) = Q.h_project(f, Q.tangent_tt(f, z), H)

function Q.h_gram(orbitals, H::CartesianHamiltonian; cfg = (;))
    Q.with_sampling_seed(0) do
        bras, kets, operators = Q.mpo_basis(orbitals, (H.diagonal, H.plus..., H.minus...))
        potential, plus, minus = operators[1], operators[2:4], operators[5:7]
        overlaps = Q.gram(orbitals; cfg)
        # Build gradients only when direct FD2 contractions lose significant digits.
        gradients = Dict{Tuple{Int,Int},Q.QTT}()
        gradient(i, a) =
            get!(gradients, (i, a)) do
                Q.shift_difference(H.plus[a], orbitals[i])
            end
        faces = ntuple(a -> axis_mpo(Q.diagonal(Q.delta_tt(H.grid.R, (1 << H.grid.R) - 1)), a), 3)
        out = zeros(ComplexF64, length(orbitals), length(orbitals))
        for j in eachindex(orbitals), i = 1:j
            value = Q.ITensorMPS.inner(bras[i], potential, kets[j]; make_inds_match = false)
            overlap = overlaps[i, j]
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
                            Q.inner_mpo(orbitals[i], faces[a], orbitals[j])
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
function green_operator(p::Problem{<:AbstractModel3D}, a, s, physics, ops)
    zero = Q.constant_tt(nsites(p.grid), 0)
    h = CartesianHamiltonian(zero, Q.diagonal(zero), ops.plus, ops.minus, p.grid, p.mass, nothing)
    CartesianGreen(
        h,
        s.green_shift,
        physics.green_relative_tolerance,
        physics.green_max_iterations,
        a.work_cap,
    )
end

"Residual of the stored state under the untruncated discrete Green operator."
function green_residual(G::CartesianGreen, psi, rhs)
    Q.linear_expression((rhs, kinetic_expression(G.kinetic, psi), psi), (1, -1, -G.shift))
end

function Q.apply_mpo(G::CartesianGreen, rhs::Q.QTT; tol = 1e-11, cap = G.cap, kwargs...)
    Q.validate_truncation(tol, cap)
    target = G.tolerance
    worktol = min(max(tol, eps(Float64)), target / 100)
    workcap = max(cap, G.cap)
    combine(xs, ws) =
        Q.materialize_expression(Q.linear_expression(xs, ws); tol = worktol, cap = workcap)
    action(x) = combine((kinetic_expression(G.kinetic, x), x), (1, G.shift))
    normrhs = Q.ttnorm(rhs)
    normrhs == 0 && return Q.constant_tt(length(rhs), 0)
    x, r, direction = Q.constant_tt(length(rhs), 0), rhs, rhs
    rr = real(Q.inner(r, r))
    relative = 1.0
    for iteration = 1:(G.max_iterations)
        ad = action(direction)
        curvature = real(Q.inner(direction, ad))
        curvature > 0 || error("3D Green CG lost positive curvature; increase work_cap")
        alpha = rr / curvature
        x = combine((x, direction), (1, alpha))
        residual = combine((r, ad), (1, -alpha))
        next_rr = real(Q.inner(residual, residual))
        relative = sqrt(max(next_rr, 0)) / normrhs
        if sqrt(max(next_rr, 0)) <= target * normrhs
            result = Q.rounded(x; tol, cap)
            expression = green_residual(G, result, rhs)
            relative = sqrt(real(only(Q.expression_gram([expression])))) / normrhs
            relative <= target && return result
            # Truncation is allowed for the next search direction, not for acceptance.
            true_residual = Q.materialize_expression(expression; tol = worktol, cap = workcap)
            x, r, direction = result, true_residual, true_residual
            rr = real(Q.inner(r, r))
        else
            direction = combine((residual, direction), (1, next_rr / rr))
            r, rr = residual, next_rr
        end
    end
    result = Q.rounded(x; tol, cap)
    expression = green_residual(G, result, rhs)
    relative = sqrt(real(only(Q.expression_gram([expression])))) / normrhs
    relative <= target && return result
    error(
        "3D Green CG residual $relative exceeds tolerance $(G.tolerance) after $(G.max_iterations) iterations; work cap=$workcap",
    )
end
Q.project(f::Q.Frame, psi::Q.QTT, G::CartesianGreen) =
    Q.project(f, Q.apply_mpo(G, psi; tol = G.tolerance / 100, cap = G.cap))

function total_energy(
    p::Problem{<:AbstractModel3D},
    orbitals,
    rho,
    ops,
    a,
    rec;
    time = 0.0,
    drive = NoDrive(),
)
    drive isa NoDrive || throw(ArgumentError("3D drives are not implemented"))
    zero = Q.constant_tt(nsites(p.grid), 0)
    kinetic_h =
        CartesianHamiltonian(zero, Q.diagonal(zero), ops.plus, ops.minus, p.grid, p.mass, nothing)
    kinetic = sum(
        Q.recorded_map(Float64, length(orbitals), rec.options, rec; random = false) do i, localrec
            p.occupations[i] * Q.h_expectation(kinetic_h, orbitals[i])
        end,
    )
    dv = volume_element(p.grid)
    external = dv * real(Q.inner(rho, ops.external))
    vh = hartree(rho, p, ops, a, rec)
    eh = dv / 2 * real(Q.inner(rho, vh))
    exc = if ops.xc_table === nothing
        0.0
    else
        e = Q.cross(
            rec,
            nsites(p.grid),
            Q.pointwise(n -> lda_pz81(real(n))[1], Q.sampled(rho, rec));
            tol = a.xc_energy_tolerance,
            cap = field_caps(a).xc_energy_cap,
            label = "xc_energy_density",
        )
        dv * real(sum(Q.tensortrain(e)))
    end
    (;
        total = kinetic + external + eh + exc + ops.nuclear,
        kinetic,
        external,
        hartree = eh,
        xc = exc,
        nuclear = ops.nuclear,
        drive = 0.0,
    )
end
