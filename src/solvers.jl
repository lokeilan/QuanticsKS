# TT tangent-space projection and reconstruction; see Rakhuba et al., Sec. 3.
# https://arxiv.org/abs/1811.11049
# left/right hold orthogonal cores U/V; variations satisfy the left gauge condition.

struct TangentSpace
    left::Vector{Array{ComplexF64,3}}
    right::Vector{Array{ComplexF64,3}}
    shapes::Vector{Tuple{Int,Int,Int}}
    offsets::Vector{Int}
end
function tangent_space(x::QTT; cfg = (;))
    # Tangent bases need orthogonality, not another rank truncation.
    u, v = with_sampling_seed(0) do
        psi = ITensorMPS.MPS(tensortrain(x))
        ITensorMPS.orthogonalize!(psi, 1)
        right = QTT(psi).c
        ITensorMPS.orthogonalize!(psi, length(x))
        QTT(psi).c, right
    end
    d = length(x)
    shapes = [(size(u[k], 1), 2, size(v[k], 3)) for k = 1:d]
    offsets = cumsum(vcat(0, [prod(s) for s in shapes]))
    TangentSpace(u, v, shapes, offsets)
end
function tangent_cores(tangent_basis::TangentSpace, z)
    [
        reshape(
            view(z, (tangent_basis.offsets[k] + 1):tangent_basis.offsets[k + 1]),
            tangent_basis.shapes[k],
        ) for k = 1:length(tangent_basis.shapes)
    ]
end
# Remove the component parallel to the left basis to avoid redundant tangent coordinates.
function left_gauge!(g, u)
    m = reshape(g, :, size(g, 3))
    q = reshape(u, :, size(u, 3))
    m .-= q * (q' * m)
    g
end
"Project an MPO action directly into tangent coordinates, without constructing W*x."
function project(tangent_basis::TangentSpace, x::QTT, w = nothing)
    # The tangent basis and trial orbital can differ, so environments carry separate bonds.
    d = length(x)
    if w === nothing
        w = [reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1) for k = 1:d]
    end
    right = Vector{Array{ComplexF64,3}}(undef, d + 1)
    right[d + 1] = ones(ComplexF64, 1, 1, 1)
    for k = d:-1:1
        v = tangent_basis.right[k]
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        @tensor t1[l, s, I, J] := z[l, s, L] * t[I, J, L]
        @tensor t2[j, s, l, I] := a[j, s, t, J] * t1[l, t, I, J]
        @tensor e[i, j, l] := conj(v[i, s, I]) * t2[j, s, l, I]
        right[k] = e
    end
    left = ones(ComplexF64, 1, 1, 1)
    variations = Array{ComplexF64,3}[]
    for k = 1:d
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        u = tangent_basis.left[k]
        @tensor t1[i, j, s, L] := left[i, j, l] * z[l, s, L]
        @tensor t2[i, s, J, L] := t1[i, j, t, L] * a[j, s, t, J]
        @tensor g[i, s, I] := t2[i, s, J, L] * t[I, J, L]
        k < d && left_gauge!(g, u)
        push!(variations, g)
        if k < d
            @tensor e[I, J, L] := conj(u[i, s, I]) * t2[i, s, J, L]
            left = e
        end
    end
    reduce(vcat, vec.(variations))
end
function tangent_vector(tangent_basis::TangentSpace, z)
    variations = tangent_cores(tangent_basis, z)
    c = Array{ComplexF64,3}[]
    d = length(variations)
    for k = 1:d
        u = tangent_basis.left[k]
        v = tangent_basis.right[k]
        a = variations[k]
        if k == 1
            g = cat(a, u; dims = 3)
        elseif k == d
            g = cat(v, a; dims = 1)
        else
            g = zeros(ComplexF64, size(v, 1) + size(u, 1), 2, size(v, 3) + size(u, 3))
            g[1:size(v, 1), :, 1:size(v, 3)] = v
            g[(size(v, 1) + 1):end, :, 1:size(v, 3)] = a
            g[(size(v, 1) + 1):end, :, (size(v, 3) + 1):end] = u
        end
        push!(c, g)
    end
    QTT(c)
end

# 1D FD2 Hamiltonian sampling, tangent projection, and exact MPO expectation values.

struct Hamiltonian{S}
    potential::QTT
    diagonal::Vector{Array{ComplexF64,4}}
    plus::Vector{Array{ComplexF64,4}}
    minus::Vector{Array{ComplexF64,4}}
    h::Float64
    samples::S
end
Hamiltonian(v, d, plus, minus, h) = Hamiltonian(v, d, plus, minus, Float64(h), nothing)

"Share the fixed local field across all sampled actions of one Hamiltonian."
potential_samples(H::Hamiltonian, rec) =
    H.samples === nothing ? memoized_sample(H.potential, rec) : H.samples
# Used only when every initial pivot cancels. Preserve a core-algebra fallback
# instead of declaring an unsampled large-grid function globally zero.
function h_core_fallback(H, psi, energy; tol, cap)
    shift_tol = min(tol, tol * H.h^2)
    terms = (
        psi,
        apply_mpo(H.diagonal, psi; tol, cap),
        apply_mpo(H.plus, psi; tol = shift_tol, cap),
        apply_mpo(H.minus, psi; tol = shift_tol, cap),
    )
    weights = (1 / H.h^2 - energy, 1, -1 / (2H.h^2), -1 / (2H.h^2))
    reconstruct_expression(linear_expression(terms, weights); tol, cap)
end

"Sample just the final FD2 residual, without fitting H*x, V*x or shifted states."
function h_residual(
    H,
    psi,
    energy,
    cfg,
    rec;
    tol = cfg.algebra_tolerance,
    cap = cfg.work_cap,
    label = "orbital_residual",
    potential = nothing,
)
    validate_truncation(tol, cap)
    isfinite(energy) || throw(ArgumentError("Nonfinite orbital energy"))
    orbital = sampled(psi, rec)
    field =
        potential === nothing ? potential_samples(H, rec) :
        potential isa SampledFunction ? potential :
        potential isa Evaluator ?
        TensorSamples(
            potential;
            tt_cache_capacity = get(rec.options, :slice_cache_capacity, 2048),
        ) : sampled(potential, length(psi))
    inverse_two_h_squared = 1 / (2H.h^2)
    fn = pointwise(
        (center, left, right, v) ->
            ((center - left) + (center - right)) * inverse_two_h_squared + (v - energy) * center,
        orbital,
        shifted(orbital, -1),
        shifted(orbital, 1),
        field,
    )
    cross(
        rec,
        length(psi),
        fn;
        tol,
        cap,
        label,
        initial_zero_fallback = () -> h_core_fallback(H, psi, energy; tol, cap),
    )
end

function h_action(H, psi, cfg, rec; label = "", work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid H-action tolerance"))
    tol = work_tol > 0 ? work_tol : cfg.algebra_tolerance
    h_residual(
        H,
        psi,
        0.0,
        cfg,
        rec;
        tol,
        potential,
        label = (work_tol > 0 ? "Hpsi_work_" : "Hpsi_") * label,
    )
end
function h_project(f, psi, H)
    project(f, psi, H.diagonal) +
    (project(f, psi) - (project(f, psi, H.plus) + project(f, psi, H.minus)) / 2) / H.h^2
end

"Project H on gauge-constrained tangent coordinates, such as L4's projected/SVD columns."
function h_project_tangent(f, z, H)
    t = tangent_vector(f, z)
    project(f, t, H.diagonal) + (z - (project(f, t, H.plus) + project(f, t, H.minus)) / 2) / H.h^2
end
function h_expectation(H, psi)
    # All coefficient-Hamiltonian blocks must use the same MPO contraction.
    # A loose TCI Rayleigh diagonal becomes an artificial negative direction
    # when the joint old-orbital / tangent Gram matrix is nearly singular.
    real(only(h_gram([psi], H)))
end

function shift_difference(shift, psi)
    # Exact low-rank shift, used only for cancellation-prone kinetic entries.
    # No sampled fit or finite-tolerance truncation enters this fallback.
    cap = maximum(size(w, 4) * size(g, 3) for (w, g) in zip(shift, psi.c))
    shifted = apply_mpo(shift, psi; tol = 0.0, cap)
    reconstruct_expression(
        linear_expression((shifted, psi), (1, -1));
        tol = 0.0,
        cap = rank(shifted) + rank(psi),
    )
end

"Project H from cores; use stable gradient contractions when FD2 terms cancel."
function h_gram(orbitals, H; cfg = (;))
    with_sampling_seed(0) do
        b = length(orbitals)
        bras, kets, (potential, plus, minus) = mpo_basis(orbitals, (H.diagonal, H.plus, H.minus))
        overlaps = gram(orbitals; cfg)
        A = zeros(ComplexF64, b, b)
        fallback = fill(false, b, b)
        foreach_independent(b, cfg) do j
            for i = 1:j
                terms = (
                    overlaps[i, j],
                    -ITensorMPS.inner(bras[i], plus, kets[j]; make_inds_match = false) / 2,
                    -ITensorMPS.inner(bras[i], minus, kets[j]; make_inds_match = false) / 2,
                )
                numerator = compensated_sum(terms)
                fallback[i, j] =
                    abs(numerator) <= sqrt(eps(Float64)) * sum(abs, terms) ||
                    (i == j && real(numerator) < 0)
                A[i, j] =
                    ITensorMPS.inner(bras[i], potential, kets[j]; make_inds_match = false) +
                    (fallback[i, j] ? 0 : numerator / H.h^2)
            end
        end
        needed = [i for i = 1:b if any(fallback[min(i, j), max(i, j)] for j = 1:b)]
        gradients = Vector{QTT}(undef, b)
        foreach_independent(length(needed), cfg) do k
            i = needed[k]
            gradients[i] = shift_difference(H.plus, orbitals[i])
        end
        foreach_independent(b, cfg) do j
            for i = 1:j
                fallback[i, j] && (A[i, j] += inner(gradients[i], gradients[j]; cfg) / (2H.h^2))
                A[j, i] = conj(A[i, j])
                i == j && (A[i, i] = real(A[i, i]))
            end
        end
        require_finite(A, "nonfinite_direct_hamiltonian_gram")
        A
    end
end

# Orbital updates, orthogonalization, and density mixing.

# Keep consecutive orbital transforms in coefficient space until cores are needed.

mutable struct OrbitalTransform
    basis::Vector{QTT}
    coefficients::Matrix{ComplexF64}
    metric::Matrix{ComplexF64}
    transforms::Int
    rotated::Bool
end
function OrbitalTransform(orbitals; cfg = (;), metric = gram(orbitals; cfg))
    OrbitalTransform(
        collect(orbitals),
        Matrix{ComplexF64}(I, length(orbitals), length(orbitals)),
        metric,
        0,
        false,
    )
end
transform_gram(x::OrbitalTransform) = x.coefficients' * x.metric * x.coefficients

function orthogonalize_coefficients!(x::OrbitalTransform, cfg; force = false)
    S = Hermitian(transform_gram(x))
    require_finite(Matrix(S), "nonfinite_orbital_transform_gram")
    defect = norm(S - I)
    defect < cfg.orthogonalize_threshold && !force && return false
    fac = eigen(S)
    minimum(fac.values) > cfg.independence_threshold ||
        error("Occupied orbitals lost linear independence")
    x.coefficients =
        x.coefficients * (fac.vectors * Diagonal(1 ./ sqrt.(fac.values)) * fac.vectors')
    x.transforms += 1
    true
end

function ritz_projection(orbitals, H, cfg, rec; work_tol = 0.0, potential = nothing)
    stage = work_tol > 0 ? "ritz_sampled_projection" : "ritz_direct_projection"
    timed(rec, stage) do
        work_tol > 0 || return h_gram(orbitals, H; cfg)
        # Apply H once per input column, at the existing work tolerance/cap.
        # The following coefficient transforms reuse this fixed-basis matrix.
        field = synchronized(potential === nothing ? potential_samples(H, rec) : potential)
        hx = recorded_map(QTT, length(orbitals), cfg, rec) do i, localrec
            h_action(H, orbitals[i], cfg, localrec; label = "ritz_$i", work_tol, potential = field)
        end
        A = gram(orbitals, hx; cfg)
        require_finite(A, "nonfinite_sampled_ritz_projection")
        (A + A') / 2
    end
end
function ritz_coefficients!(
    x::OrbitalTransform,
    H,
    cfg,
    rec;
    force = false,
    work_tol = 0.0,
    potential = nothing,
)
    projected = ritz_projection(x.basis, H, cfg, rec; work_tol, potential)
    A = x.coefficients' * projected * x.coefficients
    A = (A + A') / 2
    require_finite(A, "nonfinite_orbital_transform_hamiltonian")
    offdiag = norm(A - Diagonal(diag(A)))
    !force && offdiag < cfg.eigen_tolerance / 10 && return offdiag
    x.coefficients = x.coefficients * eigen(Hermitian(A)).vectors
    squared_norms = real.(diag(transform_gram(x)))
    all(n -> isfinite(n) && n > 0, squared_norms) || error("Invalid rotated orbital norm")
    x.coefficients ./= transpose(sqrt.(squared_norms))
    x.transforms += 1
    x.rotated = true
    offdiag
end

function reconstruct_orbitals(x::OrbitalTransform, cfg, rec; label)
    evaluator = synchronized(block_evaluator(x.basis, rec))
    group_size = get(rec.options, :orbital_pivot_group_size, 4)
    updated_orbitals =
        recorded_map(QTT, size(x.coefficients, 2), cfg, rec; group_size) do i, localrec
            y = linear_combination(
                localrec,
                x.basis,
                view(x.coefficients, :, i);
                evaluator,
                tol = cfg.orthogonalization_tolerance,
                cap = cfg.orbital_cap,
                label = "$(label)_$i",
                shared_pivots = "orbital_basis_group_$(cld(i, group_size))",
            )
            x.rotated ? normalize(y) : y
        end
    updated_orbitals, block_cache_stats(evaluator)
end

function finish_orbital_transform(x::OrbitalTransform, original, cfg, rec)
    x.transforms == 0 && return original, norm(x.metric - I)
    label = x.rotated ? "ritz_rotate" : "orthogonalize"
    updated_orbitals, cache_stats = reconstruct_orbitals(x, cfg, rec; label)
    metric = gram(updated_orbitals; cfg)
    defect = norm(metric - I)
    repaired = false
    # Compression is nonlinear: a coefficient-space identity does not certify
    # the fitted cores. Retain the original one-pass post-Ritz repair if needed.
    if x.rotated && defect >= cfg.orthogonalize_threshold
        updated_orbitals, defect = orthogonalize(updated_orbitals, cfg, rec; metric)
        repaired = true
    end
    event!(
        rec.stage_io,
        (;
            scf = rec.scf,
            inner = rec.inner,
            stage = "orbital_transform",
            parent_stage = rec.stage,
            seconds = 0.0,
            gc_seconds = 0.0,
            allocated_bytes = 0,
            tci_calls = 0,
            tci_samples = 0,
            tci_seconds = 0.0,
            event = "orbital_transform",
            basis_columns = length(x.basis),
            coefficient_transforms = x.transforms,
            reconstruction_passes = 1 + Int(repaired),
            post_fit_orthogonalization = repaired,
            gram_defect = defect,
            cache_stats...,
        ),
    )
    updated_orbitals, defect
end

function orthogonalize(orbitals, cfg, rec; force = false, metric = gram(orbitals; cfg))
    transform = OrbitalTransform(orbitals; cfg, metric)
    orthogonalize_coefficients!(transform, cfg; force)
    finish_orbital_transform(transform, orbitals, cfg, rec)
end

function ritz_rotate(orbitals, H, cfg, rec; work_tol = 0.0, force = false)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(orbitals; cfg)
    offdiag = ritz_coefficients!(transform, H, cfg, rec; force, work_tol)
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    updated_orbitals, _ = finish_orbital_transform(transform, orbitals, cfg, rec)
    updated_orbitals, offdiag
end

"Orthogonalize, optionally rotate, and rebuild only the final occupied columns."
function orthogonalize_ritz(orbitals, H, cfg, rec; ritz = true, work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(orbitals; cfg)
    orthogonalize_coefficients!(transform, cfg)
    offdiag = ritz ? ritz_coefficients!(transform, H, cfg, rec; work_tol, potential) : 0.0
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    updated_orbitals, defect = finish_orbital_transform(transform, orbitals, cfg, rec)
    updated_orbitals, defect, offdiag
end

# Optimize small coefficient matrices while enforcing orthogonality between orbitals.

"Alternating minimization with diagonal coefficients on old orbitals (Rakhuba et al., Eqs. (23)-(29))."
function coefficients(
    S_xx,
    S_vx,
    H_vx,
    H_vv,
    energies;
    nsweeps = 2,
    parallel_tasks = 1,
    metric_relative_cutoff = 1e-11,
    metric_absolute_cutoff = 1e-13,
    constraint_relative_cutoff = 1e-10,
    constraint_absolute_cutoff = 1e-12,
)
    all(isfinite, S_xx) &&
        all(isfinite, S_vx) &&
        all(isfinite, H_vx) &&
        all(isfinite, H_vv) &&
        all(isfinite, energies) || error("Nonfinite input to coefficient minimizer")
    # X contains orbitals and V orthonormal tangent directions: S_vx = V'X, H_vx = V'HX.
    # S_xx = X'X and H_vv = V'HV; c and C follow the paper's coefficient notation.
    m, b = size(S_vx)
    G = [S_xx S_vx'; S_vx Matrix{ComplexF64}(I, m, m)]
    coef = vcat(Matrix{ComplexF64}(I, b, b), zeros(ComplexF64, m, b))
    orthogonalizers = Vector{Matrix{ComplexF64}}(undef, b)
    hamiltonians = Vector{Matrix{ComplexF64}}(undef, b)
    foreach_independent(b, (; parallel_tasks)) do i
        local ids, a
        ids = vcat(i, (b + 1):(b + m))
        g = Hermitian(G[ids, ids])
        e = eigen(g)
        good = findall(
            >(max(maximum(e.values) * metric_relative_cutoff, metric_absolute_cutoff)),
            e.values,
        )
        orthogonalizers[i] = e.vectors[:, good] * Diagonal(1 ./ sqrt.(e.values[good]))
        a = zeros(ComplexF64, m + 1, m + 1)
        a[1, 1] = energies[i] * real(S_xx[i, i])
        a[2:end, 1] = H_vx[:, i]
        a[1, 2:end] = conj.(H_vx[:, i])
        a[2:end, 2:end] = H_vv
        hamiltonians[i] = a
    end
    # Each orbital's constraints use the coefficients already updated in this sweep.
    for sweep = 1:nsweeps, i = 1:b
        ids = vcat(i, (b + 1):(b + m))
        other = [j for j = 1:b if j != i]
        orthogonalizer = orthogonalizers[i]
        constraints = orthogonalizer' * (G[ids, :] * coef[:, other])
        s = checked_svd(constraints; full = true)
        constraint_rank = count(
            >(
                max(
                    isempty(s.S) ? 0.0 : maximum(s.S) * constraint_relative_cutoff,
                    constraint_absolute_cutoff,
                ),
            ),
            s.S,
        )
        null_basis = s.U[:, (constraint_rank + 1):end]
        size(null_basis, 2) > 0 || error("No feasible coefficient direction")
        feasible_basis = orthogonalizer * null_basis
        a = Hermitian(feasible_basis' * hamiltonians[i] * feasible_basis)
        all(isfinite, a) || error(
            "Nonfinite reduced coefficient matrix at sweep=$sweep orbital=$i; orthogonalizer_max=$(maximum(abs,orthogonalizer)) nullspace_size=$(size(null_basis))",
        )
        ev = eigen(a)
        v = feasible_basis * ev.vectors[:, 1]
        overlap = dot(G[ids, i], v)
        abs(overlap) > 1e-12 && (v .*= conj(overlap) / abs(overlap))
        coef[:, i] .= 0
        coef[ids, i] = v
    end
    trace_after = sum(
        real(
            dot(
                coef[vcat(i, (b + 1):(b + m)), i],
                hamiltonians[i] * coef[vcat(i, (b + 1):(b + m)), i],
            ),
        ) for i = 1:b
    )
    (
        c = diag(coef[1:b, :]),
        C = coef[(b + 1):end, :],
        gram_defect = norm(coef' * G * coef - I),
        trace_after = trace_after,
    )
end

# Low-rank orbital update; stages L1 to L7 match the performance logs.

"Green fixed-point update: (T + mu*I)^(-1) * (mu + epsilon - V) * psi."
function block_green(orbitals, energies, H, green, cfg, rec; potential = nothing)
    potential_samples_shared =
        synchronized(potential === nothing ? potential_samples(H, rec) : potential)
    recorded_map(QTT, length(orbitals), cfg, rec) do i, localrec
        psi_samples = sampled(orbitals[i], localrec)
        rhs = cross(
            localrec,
            cfg.R,
            pointwise(
                (v, value) -> (cfg.green_shift + energies[i] - v) * value,
                potential_samples_shared,
                psi_samples,
            );
            tol = cfg.green_rhs_tolerance,
            cap = get(cfg, :green_rhs_cap, cfg.work_cap),
            label = "block_green_$i",
        )
        # Truncated MPO action establishes its own QR gauge when needed.
        normalize(apply_mpo(green, rhs; tol = cfg.algebra_tolerance, cap = cfg.orbital_cap))
    end
end
function lrrap_step(orbitals, previous, H, green, cfg, rec; work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 ||
        throw(ArgumentError("work_tol must be finite and nonnegative"))
    algebra_tol = work_tol > 0 ? work_tol : cfg.algebra_tolerance
    state_tol = min(algebra_tol, cfg.orbital_tolerance)
    b = length(orbitals)
    field = potential === nothing ? potential_samples(H, rec) : potential
    # L1: contract Rayleigh energies and sample only the final residuals.
    energies, residuals = timed(rec, "L1") do
        potential_samples_shared = synchronized(field)
        results = recorded_map(Tuple{Float64,QTT}, b, cfg, rec) do i, localrec
            e = h_expectation(H, orbitals[i]) / real(inner(orbitals[i], orbitals[i]))
            r = h_residual(
                H,
                orbitals[i],
                e,
                cfg,
                localrec;
                tol = algebra_tol,
                cap = cfg.work_cap,
                label = "orbital_residual_$i",
                potential = potential_samples_shared,
            )
            e, r
        end
        first.(results), last.(results)
    end
    residual_norms = independent_map(i -> ttnorm(residuals[i]), Float64, b, cfg)
    # Cycle the tangent-space base point through the occupied orbitals.
    tangent_orbital = mod((rec.scf - 1) * cfg.inner_steps + rec.inner - 1, b) + 1
    tangent_basis = timed(rec, "L2") do
        tangent_space(orbitals[tangent_orbital]; cfg)
    end
    # L3: project orbitals, Green-preconditioned residuals, and prior directions into one tangent space.
    DX, DZ, DP = timed(rec, "L3") do
        rows = tangent_basis.offsets[end]
        dx = independent_columns(i -> project(tangent_basis, orbitals[i]), rows, b, cfg)
        dz = independent_columns(i -> project(tangent_basis, residuals[i], green), rows, b, cfg)
        dp = independent_columns(
            i -> project(tangent_basis, previous[i]),
            rows,
            previous === nothing ? 0 : length(previous),
            cfg,
        )
        dx, dz, dp
    end
    # Only tangent coordinates are needed beyond this point.
    empty!(residuals)
    D, W, H_vv, S_vx, H_vx, S_xx = timed(rec, "L4") do
        raw = hcat(DX, DZ, DP)
        # Orthonormalize the tangent coordinates with SVD. X retains diagonal coefficients.
        column_norms = [max(norm(view(raw, :, j)), 1e-14) for j = 1:size(raw, 2)]
        raw ./= transpose(column_norms)
        # Overwrite the normalized coordinate buffer and retain a view of U.
        # This avoids three extra copies of the largest tangent-coordinate array.
        fac = svd!(raw; alg = LinearAlgebra.QRIteration())
        good = findall(
            >(max(maximum(fac.S) * cfg.tangent_relative_cutoff, cfg.tangent_absolute_cutoff)),
            fac.S,
        )
        W = Diagonal(1 ./ column_norms) * fac.V[:, good] * Diagonal(1 ./ fac.S[good])
        D = view(fac.U, :, 1:length(good))
        H_vv = independent_columns(size(D, 2), size(D, 2), cfg) do j
            hv = h_project_tangent(tangent_basis, view(D, :, j), H)
            D' * hv
        end
        H_vv = (H_vv + H_vv') / 2
        # Exact contractions with H, independently of the TCI approximation used in L1.
        phx = independent_columns(i -> h_project(tangent_basis, orbitals[i], H), size(D, 1), b, cfg)
        D, W, H_vv, D' * DX, D' * phx, gram(orbitals; cfg)
    end
    # L5: optimize the small coefficient problem under the orbital orthogonality constraints.
    coefficient_solution = timed(rec, "L5") do
        debug_snapshot(
            rec,
            "coefficient_debug.jls",
            (S_xx = S_xx, S_vx = S_vx, H_vx = H_vx, H_vv = H_vv, energies = energies),
        )
        coefficients(
            S_xx,
            S_vx,
            H_vx,
            H_vv,
            energies;
            parallel_tasks = get(cfg, :parallel_tasks, 1),
            nsweeps = cfg.coefficient_sweeps,
            metric_relative_cutoff = cfg.metric_relative_cutoff,
            metric_absolute_cutoff = cfg.metric_absolute_cutoff,
            constraint_relative_cutoff = cfg.constraint_relative_cutoff,
            constraint_absolute_cutoff = cfg.constraint_absolute_cutoff,
        )
    end
    green_update =
        (cfg.green_update_interval > 0 && rec.scf % cfg.green_update_interval == 0) ||
        maximum(residual_norms) < cfg.green_update_residual_threshold
    # L6: reconstruct orbitals. A Green update resets the previous search directions.
    updated_orbitals, search_directions = timed(rec, "L6") do
        correction_coordinates = D * coefficient_solution.C
        history_coordinates = if green_update
            nothing
        else
            basis_coefficients = W * coefficient_solution.C
            directions = DZ * basis_coefficients[(b + 1):(2b), :]
            previous !== nothing && (directions .+= DP * basis_coefficients[(2b + 1):end, :])
            directions
        end
        results = recorded_map(Tuple{QTT,QTT}, b, cfg, rec) do i, localrec
            t = tangent_vector(tangent_basis, view(correction_coordinates, :, i))
            y = normalize(
                add(
                    orbitals[i],
                    t;
                    a = coefficient_solution.c[i],
                    tol = state_tol,
                    cap = cfg.orbital_cap,
                    rec = localrec,
                    label = "orbital_update_$i",
                ),
            )
            p = if green_update
                constant_tt(cfg.R, 0.0)
            else
                with_svd_context(;
                    orbital = i,
                    role = "history_direction",
                    coordinates = view(history_coordinates, :, i),
                ) do
                    rounded(
                        tangent_vector(tangent_basis, view(history_coordinates, :, i));
                        tol = algebra_tol,
                        cap = cfg.orbital_cap,
                    )
                end
            end
            y, p
        end
        first.(results), last.(results)
    end
    if green_update
        # Accepted orbitals keep the original tight Green output accuracy;
        # the looser work budget applies to search directions, not this state.
        updated_orbitals = timed(rec, "L6G") do
            block_green(updated_orbitals, energies, H, green, cfg, rec; potential = field)
        end
    end
    updated_orbitals, gramdef, ritz_offdiag = timed(rec, "L7") do
        orthogonalize_ritz(
            updated_orbitals,
            H,
            cfg,
            rec;
            ritz = green_update,
            work_tol,
            potential = field,
        )
    end
    metrics = (
        tangent_orbital = tangent_orbital,
        tangent_dimension = size(D, 2),
        residual_max = maximum(residual_norms),
        residual_rms = sqrt(sum(abs2, residual_norms) / b),
        trace_before = sum(energies),
        trace_after = coefficient_solution.trace_after,
        coefficient_gram_defect = coefficient_solution.gram_defect,
        gram_defect = gramdef,
        green_update = green_update,
        ritz_offdiagonal_before = ritz_offdiag,
        max_rank = maximum(rank.(updated_orbitals)),
        inner_output_max_rank = maximum(rank.(updated_orbitals)),
        orbital_bytes = sum(storage, updated_orbitals),
        history_bytes = sum(storage, search_directions),
        work_tolerance = work_tol,
        work_rank_cap = cfg.work_cap,
    )
    updated_orbitals, search_directions, metrics
end

# Density mixing and output compression use separate working and storage tolerances.

function mix_density(previous_density, orbital_density, inputs, residuals, cfg, rec)
    previous_density === nothing && return orbital_density, "initial"
    residual = linear_expression((orbital_density, previous_density), (1, -1))
    push!(inputs, previous_density)
    push!(residuals, residual)
    length(inputs) > cfg.anderson_history && (popfirst!(inputs); popfirst!(residuals))
    m = length(inputs)
    coeff = zeros(m)
    coeff[end] = 1
    method = "linear"
    if cfg.anderson && m >= 3
        B = timed(rec, "mixing_residual_gram") do
            real.(expression_gram(residuals; rec))
        end
        B .+= Diagonal(fill(maximum(diag(B)) * cfg.anderson_regularization + 1e-20, m))
        trial = B \ ones(m)
        trial ./= sum(trial)
        if all(isfinite, trial) && maximum(abs, trial) < cfg.anderson_coefficient_limit
            coeff = trial
            method = "anderson"
        end
    end
    active = findall(c -> abs(c) >= 1e-15, coeff)
    terms = vcat(inputs[active], residuals[active])
    weights = vcat(coeff[active], cfg.mixing .* coeff[active])
    # Flatten n_i + alpha*(rho_i-n_i); history holds only original QTT
    # terms, so expression depth stays one and at most 2*history terms live.
    linear_expression(terms, weights), method
end

reconstruct_density(rho::QTT, cfg, rec) = rho
function reconstruct_density(rho::QTTExpression, cfg, rec)
    timed(rec, "density_mixing") do
        mixed = reconstruct_expression(
            rho;
            tol = get(cfg, :mixing_tolerance, cfg.algebra_tolerance),
            cap = get(cfg, :density_cap, cfg.effective_potential_cap),
        )
        dx = get(cfg, :volume_element, cfg.cell / 2^cfg.R)
        charge = dx * real(sum(tensortrain(mixed)))
        isfinite(charge) && charge > 0 || error("Invalid charge in mixed density: $charge")
        scale(mixed, cfg.electrons / charge)
    end
end

"Compress stored orbitals at their requested accuracy; internal operator algebra keeps its tighter tolerance."
function compress_orbitals(orbitals, cfg; tol = cfg.orbital_tolerance, input_gram_defect = nothing)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("orbital_tol must be finite and nonnegative"))
    before = sum(storage, orbitals)
    input_defect = input_gram_defect === nothing ? norm(gram(orbitals; cfg) - I) : input_gram_defect
    tol == 0 && return orbitals,
    (
        orbital_storage_tolerance = 0.0,
        orbital_rank_before_compression = maximum(rank.(orbitals)),
        stored_max_rank = maximum(rank.(orbitals)),
        orbital_bytes_before_compression = before,
        orbital_compression_attempts = 0,
        stored_gram_defect = input_defect,
    )
    for attempt = 1:(cfg.storage_compression_attempts)
        actual = tol / 10^(attempt - 1)
        updated_orbitals = Vector{QTT}(undef, length(orbitals))
        foreach_independent(length(orbitals), cfg) do i
            updated_orbitals[i] =
                normalize(rounded(orbitals[i]; tol = actual, cap = cfg.orbital_cap))
        end
        # Compression must not sacrifice occupied-space orthogonality.
        defect = norm(gram(updated_orbitals; cfg) - I)
        if defect <= max(cfg.storage_gram_tolerance, 2input_defect)
            return updated_orbitals,
            (
                orbital_storage_tolerance = actual,
                orbital_rank_before_compression = maximum(rank.(orbitals)),
                stored_max_rank = maximum(rank.(updated_orbitals)),
                orbital_bytes_before_compression = before,
                orbital_compression_attempts = attempt,
                stored_gram_defect = defect,
            )
        end
    end
    error("Orbital compression failed the Gram check at requested tolerances")
end

# VectorInterface keeps solver tolerances and rank caps separate from the stored QTT type.
struct QTTVector
    psi::QTT
    tol::Float64
    cap::Int
end
VI.scalartype(::Type{QTTVector}) = ComplexF64
VI.inner(x::QTTVector, y::QTTVector) = inner(x.psi, y.psi)
LinearAlgebra.norm(x::QTTVector) = ttnorm(x.psi)
VI.zerovector(x::QTTVector, ::Type{<:Number}) =
    QTTVector(constant_tt(length(x.psi), 0), x.tol, x.cap)
VI.zerovector!!(x::QTTVector) = VI.zerovector(x)
VI.scale(x::QTTVector, a::Number) = QTTVector(scale(x.psi, a), x.tol, x.cap)
VI.scale!!(x::QTTVector, a::Number) = VI.scale(x, a)
VI.scale!!(::QTTVector, x::QTTVector, a::Number) = VI.scale(x, a)
function VI.add(y::QTTVector, x::QTTVector, a::Number, b::Number)
    iszero(a) && return VI.scale(y, b)
    iszero(b) && return VI.scale(x, a)
    # Use TT addition for recurrence algebra, with exact contractions for overlaps.
    psi = add(y.psi, x.psi; a = b, b = a, tol = y.tol, cap = y.cap)
    QTTVector(psi, y.tol, y.cap)
end
VI.add!!(y::QTTVector, x::QTTVector, a::Number, b::Number) = VI.add(y, x, a, b)

"Advance a scalar or block Lanczos factorization and extract its lowest Ritz orbitals."
function lanczos_step(orbitals, subspace, H, cfg, rec; work_tol, ritz_tol, potential)
    norbitals = length(orbitals)
    limit = min(2^cfg.R, max(16, 4norbitals))
    width =
        subspace === nothing ? norbitals :
        norbitals == 1 ? 1 : length(KrylovKit.residual(subspace.factorization))
    residual_floor = max(work_tol, 100eps(Float64))
    breakdown =
        subspace !== nothing &&
        KrylovKit.normres(subspace.factorization) <= sqrt(width) * residual_floor
    restarted = subspace !== nothing && !breakdown && length(subspace.factorization) + width > limit
    initialized = subspace === nothing || restarted
    if initialized
        action =
            x -> QTTVector(
                h_action(H, x.psi, cfg, rec; label = "lanczos_action", work_tol, potential),
                work_tol,
                cfg.work_cap,
            )
        initial = [QTTVector(psi, work_tol, cfg.work_cap) for psi in orbitals]
        # A block preserves independent directions within degenerate eigenspaces.
        iterator = if norbitals == 1
            KrylovKit.LanczosIterator(action, only(initial), KrylovKit.ModifiedGramSchmidt2())
        else
            KrylovKit.BlockLanczosIterator(
                action,
                KrylovKit.Block(initial),
                limit,
                KrylovKit.ModifiedGramSchmidt2(),
                residual_floor,
            )
        end
        factorization = KrylovKit.initialize(iterator)
        length(factorization) >= norbitals || error("The initial Lanczos block lost occupied rank")
        subspace = (; iterator, factorization, gram_defect = Inf)
    end
    factorization = subspace.factorization
    previous_dimension = length(factorization)
    width = norbitals == 1 ? 1 : length(KrylovKit.residual(factorization))
    # Do not normalize a numerically zero residual at an invariant subspace.
    breakdown_tolerance = sqrt(width) * residual_floor
    breakdown = KrylovKit.normres(factorization) <= breakdown_tolerance
    if !breakdown && previous_dimension + width <= limit
        KrylovKit.expand!(subspace.iterator, factorization)
    end
    dimension = length(factorization)
    basis = [x.psi for x in KrylovKit.basis(factorization)]
    if initialized || dimension > previous_dimension
        projected = if norbitals == 1
            KrylovKit.rayleighquotient(factorization) # Real symmetric tridiagonal matrix.
        else
            Hermitian(view(factorization.H, 1:dimension, 1:dimension)) # Block tridiagonal.
        end
        spectrum = eigen(projected)
        orbitals = combine_orbitals(
            basis,
            spectrum.vectors[:, 1:norbitals],
            cfg,
            rec;
            tol = ritz_tol,
            cap = cfg.orbital_cap,
            label = "lanczos_ritz",
        )
        # Repair only the returned orbitals; the retained recurrence basis must stay fixed.
        orbitals, gram_defect = orthogonalize(orbitals, cfg, rec)
        subspace = (; subspace.iterator, factorization, gram_defect)
    end
    width = norbitals == 1 ? 1 : length(KrylovKit.residual(factorization))
    breakdown = KrylovKit.normres(factorization) <= sqrt(width) * residual_floor
    metrics = (;
        solver = :lanczos,
        recurrence = norbitals == 1 ? :scalar : :block,
        trial_dimension = dimension,
        appended_directions = dimension - previous_dimension,
        restarted,
        breakdown,
        recurrence_residual = KrylovKit.normres(factorization),
        gram_defect = subspace.gram_defect,
        work_tolerance = work_tol,
        work_rank_cap = cfg.work_cap,
        max_trial_bond = maximum(rank, basis),
    )
    orbitals, subspace, metrics
end

# Davidson subspaces share the same projection and reconstruction operations.

"Canonical orthogonalization: discard small overlap eigenvalues and form W with W' * S * W = I."
function canonical_orthogonalizer(overlap, relative_cutoff, absolute_cutoff)
    metric = eigen(Hermitian((overlap + overlap') / 2))
    cutoff = max(absolute_cutoff, relative_cutoff * last(metric.values))
    independent = findall(>(cutoff), metric.values)
    metric.vectors[:, independent] * Diagonal(inv.(sqrt.(metric.values[independent])))
end

"Fit each final orbital from one shared basis cache."
function combine_orbitals(basis, coefficients, cfg, rec; tol, cap, label)
    evaluator = synchronized(block_evaluator(basis, rec))
    recorded_map(QTT, size(coefficients, 2), cfg, rec) do i, localrec
        linear_combination(
            localrec,
            basis,
            view(coefficients, :, i);
            evaluator,
            tol,
            cap,
            label = "$(label)_$i",
        )
    end
end

function subspace_ritz(basis, keep, project_matrices, cfg, rec; tol)
    hamiltonian, overlap = timed(() -> project_matrices(basis), rec, "subspace_projection")
    orthogonalizer =
        canonical_orthogonalizer(overlap, cfg.metric_relative_cutoff, cfg.metric_absolute_cutoff)
    size(orthogonalizer, 2) >= keep || error("The trial subspace lost occupied rank")
    projected = orthogonalizer' * hamiltonian * orthogonalizer
    spectrum = eigen(Hermitian((projected + projected') / 2))
    coefficients = orthogonalizer * spectrum.vectors[:, 1:keep]
    orbitals = combine_orbitals(
        basis,
        coefficients,
        cfg,
        rec;
        tol,
        cap = cfg.orbital_cap,
        label = "subspace_ritz",
    )
    # Fitting can change the Gram matrix; repair only if the stored orbitals need it.
    orbitals, defect = orthogonalize(orbitals, cfg, rec)
    orbitals, spectrum.values[1:keep], defect, overlap
end

function extend_subspace(basis, directions, room, cfg, rec; tol, metric = gram(basis; cfg))
    (isempty(directions) || room == 0) && return QTT[]
    norms = ttnorm.(directions)
    directions = [scale(x, inv(n)) for (x, n) in zip(directions, norms) if n > 100eps(Float64)]
    isempty(directions) && return directions
    # Compression can spoil basis orthogonality; use the full Gram pseudoinverse.
    projection =
        pinv(
            Hermitian(metric);
            rtol = cfg.metric_relative_cutoff,
            atol = cfg.metric_absolute_cutoff,
        ) * gram(basis, directions; cfg)
    remainders = combine_orbitals(
        vcat(basis, directions),
        vcat(-projection, Matrix{ComplexF64}(I, length(directions), length(directions))),
        cfg,
        rec;
        tol,
        cap = cfg.work_cap,
        label = "davidson_direction",
    )
    appended = KrylovKit.OrthonormalBasis{QTTVector}()
    orth = KrylovKit.ModifiedGramSchmidt2()
    norm_floor = max(tol, 100eps(Float64))
    for remainder in remainders
        n = ttnorm(remainder)
        n > norm_floor || continue
        vector = QTTVector(scale(remainder, inv(n)), tol, cfg.work_cap)
        # Test dependence within the new block after removing the retained space.
        vector, _ = KrylovKit.orthogonalize!!(vector, appended, orth)
        n = norm(vector)
        n^2 > max(cfg.metric_relative_cutoff, cfg.metric_absolute_cutoff) || continue
        push!(appended, VI.scale!!(vector, inv(n)))
        length(appended) == room && break
    end
    [x.psi for x in appended]
end

function davidson_directions(solver, orbitals, energies, H, green, cfg, rec, potential; tol)
    residuals = recorded_map(QTT, length(orbitals), cfg, rec) do i, localrec
        h_residual(
            H,
            orbitals[i],
            energies[i],
            cfg,
            localrec;
            tol,
            cap = cfg.work_cap,
            label = "davidson_residual_$i",
            potential,
        )
    end
    norms = ttnorm.(residuals)
    selected = findall(>(max(cfg.eigen_tolerance, tol, 100eps(Float64))), norms)
    solver == :davidson && !isempty(selected) && (selected = [argmax(norms)])
    candidates = independent_map(QTT, length(selected), cfg) do j
        i = selected[j]
        # Normalize the inverse's input so its work budget is independent of residual size.
        apply_mpo(green, scale(residuals[i], inv(norms[i])); tol, cap = cfg.work_cap)
    end
    candidates, maximum(norms), norm(norms) / sqrt(length(norms))
end

"Generalized Davidson: expand with preconditioned residuals and retain the lowest Ritz orbitals."
function davidson_step(
    solver,
    orbitals,
    subspace,
    H,
    green,
    cfg,
    rec;
    work_tol,
    ritz_tol,
    potential,
    project_matrices = basis -> (h_gram(basis, H; cfg), gram(basis; cfg)),
)
    solver in (:davidson, :block_davidson) || throw(ArgumentError("Unknown Davidson method"))
    norbitals = length(orbitals)
    if subspace === nothing
        basis = orbitals
        orbitals, energies, gram_defect, metric =
            subspace_ritz(basis, norbitals, project_matrices, cfg, rec; tol = ritz_tol)
        subspace = (; basis, energies, gram_defect, metric)
    end
    limit = min(2^cfg.R, max(16, 4norbitals))
    directions, residual_max, residual_rms = timed(rec, "subspace_expansion") do
        davidson_directions(
            solver,
            orbitals,
            subspace.energies,
            H,
            green,
            cfg,
            rec,
            potential;
            tol = work_tol,
        )
    end
    # Converged orbitals do not consume expansion slots or trigger a restart.
    restarted =
        length(subspace.basis) + length(directions) > limit && length(subspace.basis) > norbitals
    basis = restarted ? orbitals : subspace.basis
    metric = restarted ? gram(basis; cfg) : subspace.metric
    room = limit - length(basis)
    appended = timed(rec, "subspace_orthogonalization") do
        extend_subspace(
            basis,
            directions,
            room,
            cfg,
            rec;
            tol = min(work_tol, cfg.orthogonalization_tolerance),
            metric,
        )
    end
    if !isempty(appended)
        basis = vcat(basis, appended)
        orbitals, energies, gram_defect, metric =
            subspace_ritz(basis, norbitals, project_matrices, cfg, rec; tol = ritz_tol)
        subspace = (; basis, energies, gram_defect, metric)
    end
    metrics = (;
        solver,
        trial_dimension = length(subspace.basis),
        appended_directions = length(appended),
        restarted = restarted && !isempty(appended),
        residual_max,
        residual_rms,
        gram_defect = subspace.gram_defect,
        work_tolerance = work_tol,
        work_rank_cap = cfg.work_cap,
        max_trial_bond = maximum(rank, subspace.basis),
    )
    orbitals, subspace, metrics
end
