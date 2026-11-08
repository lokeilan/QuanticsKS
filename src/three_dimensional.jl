# Isolated 3D operators reuse the QTT algebra, TCI recorder, LRRAP, and SCF loop.

green_operator(p, a, s, physics, ops) =
    Q.green_mpo(p.grid.R, spacing(p.grid) * sqrt(p.mass), s.green_shift)

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
    tail = Matrix{ComplexF64}(undef, size(kernel.c[d], 3), 8)
    for mask = 0:7
        value = Matrix{ComplexF64}(I, size(tail, 1), size(tail, 1))
        for axis = 1:3
            value = value * kernel.c[d + axis][:, ((mask >> (axis - 1)) & 1) + 1, :]
        end
        tail[:, mask + 1] = vec(value)
    end
    out = Array{ComplexF64,4}[]
    for k = 1:d
        g = kernel.c[k]
        rl, _, rr = size(g)
        axis = mod1(k, 3)
        w = zeros(ComplexF64, k == 1 ? 1 : 8rl, 2, 2, k == d ? 1 : 8rr)
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

function restore_problem3d(d; model = nothing)
    d["boundary"] == "isolated_zero_exterior" &&
        d["units"] == "atomic" &&
        d["ordering"] == "interleaved_lsb" &&
        d["nodes"] == "cell_centers" ||
        throw(ArgumentError("Unsupported 3D checkpoint conventions"))
    md = d["model"]
    m = if md["kind"] == "all_electron_lda_3d"
        AllElectronLDA3D(md["positions"], md["charges"]; xc = Symbol(md["xc"]))
    elseif md["kind"] == "local_potential_3d" && model isa LocalPotential3D
        model
    else
        throw(ArgumentError("Supply the original LocalPotential3D for custom-potential checkpoints"))
    end
    model !== nothing && model_descriptor(model) != md && throw(ArgumentError("Model mismatch"))
    Problem(BoxGrid3D(d["R"], d["cell"]), m, d["occupations"]; mass = d["mass"])
end

function problem_from_config3d(gd, sd)
    kind = get(sd, "kind", "hydrogen_cluster")
    if kind == "hydrogen_cluster"
        require_keys(sd, ["kind", "shape", "distance", "padding", "xc"], "3D system")
        return hydrogen_cluster(;
            R = get(gd, "R", 7),
            shape = Tuple(get(sd, "shape", [3, 2, 2])),
            distance = get(sd, "distance", 2ANGSTROM_TO_BOHR),
            padding = get(sd, "padding", 10.0),
            cell = get(gd, "cell", nothing),
            xc = Symbol(get(sd, "xc", "pz81")),
        )
    elseif kind == "all_electron"
        require_keys(sd, ["kind", "positions", "charges", "occupations", "mass", "xc"], "3D system")
        m = AllElectronLDA3D(sd["positions"], sd["charges"]; xc = Symbol(get(sd, "xc", "pz81")))
        return Problem(
            BoxGrid3D(gd["R"], gd["cell"]),
            m,
            sd["occupations"];
            mass = get(sd, "mass", 1.0),
        )
    end
    throw(ArgumentError("Unsupported 3D system kind: $kind"))
end

function refine_state(
    state::OrbitalState{<:Problem{<:AbstractModel3D}},
    target::Problem;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    refinement = RefinementOptions(),
    parallel_tasks = 1,
)
    compatible_problem(state.problem, target; refinement = true) ||
        throw(ArgumentError("Refinement changes the physical problem"))
    validate(accuracy)
    validate(refinement)
    coarse = state.problem.grid
    g = target.grid
    0 < g.R - coarse.R <= refinement.max_added_bits ||
        throw(ArgumentError("Invalid 3D refinement increment"))
    rec = recorder(nothing, tci; parallel_tasks)
    cfg = solver_config(target, accuracy, SCFOptions(; parallel_tasks))
    ratio = 1 << (g.R - coarse.R)
    n = 1 << coarse.R
    nodes = collect((-(refinement.degree ÷ 2)):(refinement.degree ÷ 2 + 1))
    orbitals = Q.recorded_map(Q.QTT, length(state.orbitals), cfg, rec) do i, localrec
        ev = Q.evaluate(state.orbitals[i], localrec)
        fn =
            j -> begin
                q = grid_indices(g, j)
                u = ntuple(a -> (q[a] + 0.5) / ratio - 0.5, 3)
                base = floor.(Int, u)
                weights = ntuple(
                    a -> [
                        prod((u[a] - base[a] - b) / (v - b) for b in nodes if b != v) for v in nodes
                    ],
                    3,
                )
                value = 0.0im
                for ks in Iterators.product(eachindex(nodes), eachindex(nodes), eachindex(nodes))
                    index = ntuple(a -> base[a] + nodes[ks[a]], 3)
                    all(t -> 0 <= t < n, index) || continue
                    value += prod(weights[a][ks[a]] for a = 1:3) * ev(grid_index(coarse, index))
                end
                value / sqrt(ratio^3)
            end
        Q.cross(
            localrec,
            nsites(g),
            fn;
            tol = refinement.tolerance,
            cap = accuracy.orbital_cap,
            label = "refine_$i",
        )
    end
    orbitals = first(Q.orthogonalize(orbitals, cfg, rec))
    orbitals = first(Q.pack_orbitals(orbitals, cfg; tol = accuracy.storage_tolerance))
    OrbitalState(target, orbitals; time = state.time)
end

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

function kinetic_action(H::CartesianHamiltonian, psi; tol, cap)
    weights = Float64[sum(1 ./ (H.mass .* spacing(H.grid) .^ 2))]
    terms = Q.QTT[psi]
    for a = 1:3, w in (H.plus[a], H.minus[a])
        push!(terms, Q.apply_mpo(w, psi; tol = 0.0, cap = 2Q.rank(psi)))
        push!(weights, -1 / (2H.mass * spacing(H.grid)[a]^2))
    end
    Q.materialize_expression(Q.linear_expression(terms, weights); tol, cap)
end
function Q.h_core_fallback(H::CartesianHamiltonian, psi, energy; tol, cap)
    kinetic = kinetic_action(H, psi; tol, cap)
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
    Q.algebra_options(tol, cap)
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
                psi = orbitals[i]
                shifted = Q.apply_mpo(H.plus[a], psi; tol = 0.0, cap = 2Q.rank(psi))
                Q.materialize_expression(
                    Q.linear_expression((shifted, psi), (1, -1));
                    tol = 0.0,
                    cap = Q.rank(shifted) + Q.rank(psi),
                )
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
    h = spacing(G.kinetic.grid)
    terms = Q.QTT[rhs, psi]
    weights = Float64[1, -G.shift - sum(1 ./ (G.kinetic.mass .* h .^ 2))]
    for axis = 1:3, shift in (G.kinetic.plus[axis], G.kinetic.minus[axis])
        push!(terms, Q.apply_mpo(shift, psi; tol = 0.0, cap = 2Q.rank(psi)))
        push!(weights, 1 / (2G.kinetic.mass * h[axis]^2))
    end
    Q.linear_expression(terms, weights)
end

function Q.apply_mpo(G::CartesianGreen, rhs::Q.QTT; tol = 1e-11, cap = G.cap, kwargs...)
    Q.algebra_options(tol, cap)
    target = G.tolerance
    worktol = min(max(tol, eps(Float64)), target / 100)
    workcap = max(cap, G.cap)
    combine(xs, ws) =
        Q.materialize_expression(Q.linear_expression(xs, ws); tol = worktol, cap = workcap)
    action(x) =
        combine((kinetic_action(G.kinetic, x; tol = worktol, cap = workcap), x), (1, G.shift))
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

function initial_state(p::Problem{<:AbstractModel3D}, a, rec, guess::InitialGuess)
    validate(guess)
    option_dict(guess) == option_dict(InitialGuess()) || throw(
        ArgumentError(
            "3D initial guesses use nuclear envelopes; supply an explicit initial state for customization",
        ),
    )
    g = p.grid
    centers = p.model isa AllElectronLDA3D ? p.model.positions : [(0.0, 0.0, 0.0)]
    b = length(p.occupations)
    mode_limit = ceil(Int, cbrt(b))
    modes = sort(
        vec(collect(Iterators.product(1:mode_limit, 1:mode_limit, 1:mode_limit)));
        by = t -> sum(abs2, t),
    )
    cfg = solver_config(p, a)
    orbitals = Q.QTT[]
    for i = 1:b
        mode = modes[i]
        fn = j -> begin
            r = grid_position(g, j)
            envelope = sum(exp(-sqrt(sum(abs2, r .- c))) for c in centers)
            envelope * prod(sinpi(mode[k] * (r[k] / g.L[k] + 0.5)) for k = 1:3)
        end
        push!(
            orbitals,
            Q.normalize(
                Q.cross(
                    rec,
                    nsites(g),
                    fn;
                    tol = a.orthogonalization_tolerance,
                    cap = a.orbital_cap,
                    label = "initial_orbital_$i",
                ),
            ),
        )
    end
    OrbitalState(p, first(Q.orthogonalize(orbitals, cfg, rec)))
end

lowpass_mpo(::BoxGrid3D; kwargs...) =
    throw(ArgumentError("The low-pass interface is currently 1D only"))
fixed_spectrum(::OrbitalState{<:Problem{<:AbstractModel3D}}, ::AbstractVector{<:Real}; kwargs...) =
    throw(ArgumentError("The fixed-spectrum interface is currently 1D only"))

function observe_state(
    state::OrbitalState{<:Problem{<:AbstractModel3D}},
    ops,
    a,
    rec;
    drive = NoDrive(),
)
    p = state.problem
    S = Q.gram(state.orbitals; cfg = rec.options)
    norms = real.(diag(S))
    rho = density(p, state.orbitals, a, rec)
    energy = total_energy(p, state.orbitals, rho, ops, a, rec; time = state.time, drive)
    (;
        time = state.time,
        energy,
        electron_number = sum(p.occupations .* norms),
        norms,
        gram_defect = norm(S - I),
        max_bond_dimension = maximum(Q.rank.(state.orbitals)),
        orbital_bytes = sum(Q.storage, state.orbitals),
    )
end
