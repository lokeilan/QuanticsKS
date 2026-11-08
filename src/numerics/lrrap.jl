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
    !force && offdiag < cfg.eigen_tol / 10 && return offdiag
    x.coefficients = x.coefficients * eigen(Hermitian(A)).vectors
    squared_norms = real.(diag(transform_gram(x)))
    all(n -> isfinite(n) && n > 0, squared_norms) || error("Invalid rotated orbital norm")
    x.coefficients ./= transpose(sqrt.(squared_norms))
    x.transforms += 1
    x.rotated = true
    offdiag
end

function materialize_orbitals(x::OrbitalTransform, cfg, rec; label)
    oracle = synchronized(block_oracle(x.basis, rec))
    group_size = get(rec.options, :orbital_pivot_group_size, 4)
    updated_orbitals =
        recorded_map(QTT, size(x.coefficients, 2), cfg, rec; group_size) do i, localrec
            y = linear_combination(
                localrec,
                x.basis,
                view(x.coefficients, :, i);
                oracle,
                tol = cfg.ortho_tol,
                cap = cfg.rankcap,
                label = "$(label)_$i",
                shared_pivots = "orbital_basis_group_$(cld(i, group_size))",
            )
            x.rotated ? normalize(y) : y
        end
    updated_orbitals, block_cache_stats(oracle)
end

function finish_orbital_transform(x::OrbitalTransform, original, cfg, rec)
    x.transforms == 0 && return original, norm(x.metric - I)
    label = x.rotated ? "ritz_rotate" : "orthogonalize"
    updated_orbitals, cache_stats = materialize_orbitals(x, cfg, rec; label)
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
            materialization_passes = 1 + Int(repaired),
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

"Alternating minimization with diagonal coefficients on old orbitals (paper Eq.23-29)."
function coefficients(
    Sxx,
    Vx,
    Vhx,
    HV,
    energies;
    nsweeps = 2,
    parallel_tasks = 1,
    metric_relative_cutoff = 1e-11,
    metric_absolute_cutoff = 1e-13,
    constraint_relative_cutoff = 1e-10,
    constraint_absolute_cutoff = 1e-12,
)
    all(isfinite, Sxx) &&
        all(isfinite, Vx) &&
        all(isfinite, Vhx) &&
        all(isfinite, HV) &&
        all(isfinite, energies) || error("Nonfinite input to coefficient minimizer")
    m, b = size(Vx)
    G = [Sxx Vx'; Vx Matrix{ComplexF64}(I, m, m)]
    coef = vcat(Matrix{ComplexF64}(I, b, b), zeros(ComplexF64, m, b))
    whites = Vector{Matrix{ComplexF64}}(undef, b)
    as = Vector{Matrix{ComplexF64}}(undef, b)
    foreach_independent(b, (; parallel_tasks)) do i
        local ids, a
        ids = vcat(i, (b + 1):(b + m))
        g = Hermitian(G[ids, ids])
        e = eigen(g)
        good = findall(
            >(max(maximum(e.values) * metric_relative_cutoff, metric_absolute_cutoff)),
            e.values,
        )
        whites[i] = e.vectors[:, good] * Diagonal(1 ./ sqrt.(e.values[good]))
        a = zeros(ComplexF64, m + 1, m + 1)
        a[1, 1] = energies[i] * real(Sxx[i, i])
        a[2:end, 1] = Vhx[:, i]
        a[1, 2:end] = conj.(Vhx[:, i])
        a[2:end, 2:end] = HV
        as[i] = a
    end
    for sweep = 1:nsweeps, i = 1:b
        ids = vcat(i, (b + 1):(b + m))
        other = [j for j = 1:b if j != i]
        w = whites[i]
        constraints = w' * (G[ids, :] * coef[:, other])
        s = checked_svd(constraints; full = true)
        rr = count(
            >(
                max(
                    isempty(s.S) ? 0.0 : maximum(s.S) * constraint_relative_cutoff,
                    constraint_absolute_cutoff,
                ),
            ),
            s.S,
        )
        q = s.U[:, (rr + 1):end]
        size(q, 2) > 0 || error("No feasible coefficient direction")
        z = w * q
        a = Hermitian(z' * as[i] * z)
        all(isfinite, a) || error(
            "Nonfinite reduced coefficient matrix at sweep=$sweep orbital=$i; whiten=$(maximum(abs,w)) null=$(size(q))",
        )
        ev = eigen(a)
        v = z * ev.vectors[:, 1]
        overlap = dot(G[ids, i], v)
        abs(overlap) > 1e-12 && (v .*= conj(overlap) / abs(overlap))
        coef[:, i] .= 0
        coef[ids, i] = v
    end
    trace_after = sum(
        real(dot(coef[vcat(i, (b + 1):(b + m)), i], as[i] * coef[vcat(i, (b + 1):(b + m)), i]))
        for i = 1:b
    )
    (
        c = diag(coef[1:b, :]),
        C = coef[(b + 1):end, :],
        gram_defect = norm(coef' * G * coef - I),
        trace_after = trace_after,
    )
end

# Low-rank orbital update; stages L1 to L7 match the performance logs.

"tensorchem-style Block-Green correction; no differentiated TCI error in the RHS."
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
            tol = cfg.rhs_tol,
            cap = get(cfg, :rhs_cap, cfg.workcap),
            label = "block_green_$i",
        )
        # Truncated MPO action establishes its own QR gauge when needed.
        normalize(apply_mpo(green, rhs; tol = cfg.round_tol, cap = cfg.rankcap))
    end
end
function lrrap_step(orbitals, previous, H, green, cfg, rec; work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 ||
        throw(ArgumentError("work_tol must be finite and nonnegative"))
    algebra_tol = work_tol > 0 ? work_tol : cfg.round_tol
    state_tol = min(algebra_tol, cfg.tci_tol)
    b = length(orbitals)
    field = potential === nothing ? potential_samples(H, rec) : potential
    # L1: contract Rayleigh energies and sample only the final residuals.
    eps, residuals = timed(rec, "L1") do
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
                cap = cfg.workcap,
                label = "orbital_residual_$i",
                potential = potential_samples_shared,
            )
            e, r
        end
        first.(results), last.(results)
    end
    residual_norms = independent_map(i -> ttnorm(residuals[i]), Float64, b, cfg)
    # Sweep the tangent base over the occupied set; a converged low state must not monopolize it.
    anchor = mod((rec.scf - 1) * cfg.inner_steps + rec.inner - 1, b) + 1
    frame = timed(rec, "L2") do
        canonical_frame(orbitals[anchor]; cfg)
    end
    # L3: project orbitals, Green-preconditioned residuals, and prior directions into one tangent space.
    DX, DZ, DP = timed(rec, "L3") do
        rows = frame.offsets[end]
        dx = independent_columns(i -> project(frame, orbitals[i]), rows, b, cfg)
        dz = independent_columns(i -> project(frame, residuals[i], green), rows, b, cfg)
        dp = independent_columns(
            i -> project(frame, previous[i]),
            rows,
            previous === nothing ? 0 : length(previous),
            cfg,
        )
        dx, dz, dp
    end
    # Only tangent coordinates are needed beyond this point.
    empty!(residuals)
    D, W, HV, Vx, Vhx, Sxx = timed(rec, "L4") do
        raw = hcat(DX, DZ, DP)
        # Whiten only the shared tangent coordinates. Old X retains diagonal coefficients.
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
        HV = independent_columns(size(D, 2), size(D, 2), cfg) do j
            hv = h_project_tangent(frame, view(D, :, j), H)
            D' * hv
        end
        HV = (HV + HV') / 2
        # Exact contractions with H, independently of the TCI approximation used in L1.
        phx = independent_columns(i -> h_project(frame, orbitals[i], H), size(D, 1), b, cfg)
        D, W, HV, D' * DX, D' * phx, gram(orbitals; cfg)
    end
    # L5: optimize the small coefficient problem under the orbital orthogonality constraints.
    co = timed(rec, "L5") do
        debug_snapshot(
            rec,
            "coefficient_debug.jls",
            (Sxx = Sxx, Vx = Vx, Vhx = Vhx, HV = HV, eps = eps),
        )
        coefficients(
            Sxx,
            Vx,
            Vhx,
            HV,
            eps;
            parallel_tasks = get(cfg, :parallel_tasks, 1),
            nsweeps = cfg.coefficient_sweeps,
            metric_relative_cutoff = cfg.metric_relative_cutoff,
            metric_absolute_cutoff = cfg.metric_absolute_cutoff,
            constraint_relative_cutoff = cfg.constraint_relative_cutoff,
            constraint_absolute_cutoff = cfg.constraint_absolute_cutoff,
        )
    end
    polish =
        (cfg.green_polish_every > 0 && rec.scf % cfg.green_polish_every == 0) ||
        maximum(residual_norms) < cfg.green_polish_residual
    # L6: rebuild orbitals; polish discards history, so it needs no direction compression.
    updated_orbitals, search_directions = timed(rec, "L6") do
        corr = D * co.C
        pcorr = if polish
            nothing
        else
            rawC = W * co.C
            directions = DZ * rawC[(b + 1):(2b), :]
            previous !== nothing && (directions .+= DP * rawC[(2b + 1):end, :])
            directions
        end
        results = recorded_map(Tuple{QTT,QTT}, b, cfg, rec) do i, localrec
            t = tangent_tt(frame, view(corr, :, i))
            y = normalize(
                add(
                    orbitals[i],
                    t;
                    a = co.c[i],
                    tol = state_tol,
                    cap = cfg.rankcap,
                    rec = localrec,
                    label = "orbital_update_$i",
                ),
            )
            p = if polish
                constant_tt(cfg.R, 0.0)
            else
                with_svd_context(;
                    orbital = i,
                    role = "history_direction",
                    coordinates = view(pcorr, :, i),
                ) do
                    rounded(tangent_tt(frame, view(pcorr, :, i)); tol = algebra_tol, cap = cfg.rankcap)
                end
            end
            y, p
        end
        first.(results), last.(results)
    end
    if polish
        # Accepted orbitals keep the original tight Green output accuracy;
        # the looser work budget applies to search directions, not this state.
        updated_orbitals = timed(rec, "L6G") do
            block_green(updated_orbitals, eps, H, green, cfg, rec; potential = field)
        end
    end
    updated_orbitals, gramdef, ritz_offdiag = timed(rec, "L7") do
        orthogonalize_ritz(updated_orbitals, H, cfg, rec; ritz = polish, work_tol, potential = field)
    end
    metrics = (
        anchor = anchor,
        tangent_dimension = size(D, 2),
        residual_max = maximum(residual_norms),
        residual_rms = sqrt(sum(abs2, residual_norms) / b),
        trace_before = sum(eps),
        trace_after = co.trace_after,
        coefficient_gram_defect = co.gram_defect,
        gram_defect = gramdef,
        green_polish = polish,
        ritz_offdiagonal_before = ritz_offdiag,
        max_rank = maximum(rank.(updated_orbitals)),
        inner_output_max_rank = maximum(rank.(updated_orbitals)),
        orbital_bytes = sum(storage, updated_orbitals),
        history_bytes = sum(storage, search_directions),
        work_tolerance = work_tol,
        work_rank_cap = cfg.workcap,
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
    # leaves, so expression depth stays one and at most 2*history terms live.
    linear_expression(terms, weights), method
end

materialize_density(rho::QTT, cfg, rec) = rho
function materialize_density(rho::QTTExpression, cfg, rec)
    timed(rec, "density_mixing") do
        mixed = materialize_expression(
            rho;
            tol = get(cfg, :mixing_tol, cfg.round_tol),
            cap = get(cfg, :density_cap, cfg.opcap),
        )
        dx = get(cfg, :volume_element, cfg.cell / 2^cfg.R)
        charge = dx * real(sum(tensortrain(mixed)))
        isfinite(charge) && charge > 0 || error("Invalid charge in mixed density: $charge")
        scale(mixed, cfg.electrons / charge)
    end
end

"Compress stored orbitals at their requested accuracy; internal operator algebra keeps its tighter tolerance."
function pack_orbitals(orbitals, cfg; tol = cfg.tci_tol, input_gram_defect = nothing)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("orbital_tol must be finite and nonnegative"))
    before = sum(storage, orbitals)
    input_defect = input_gram_defect === nothing ? norm(gram(orbitals; cfg) - I) : input_gram_defect
    tol == 0 && return orbitals,
    (
        orbital_storage_tolerance = 0.0,
        orbital_rank_before_pack = maximum(rank.(orbitals)),
        stored_max_rank = maximum(rank.(orbitals)),
        orbital_bytes_before_pack = before,
        orbital_pack_attempts = 0,
        packed_gram_defect = input_defect,
    )
    for attempt = 1:(cfg.pack_attempts)
        actual = tol / 10^(attempt - 1)
        updated_orbitals = Vector{QTT}(undef, length(orbitals))
        foreach_independent(length(orbitals), cfg) do i
            updated_orbitals[i] = normalize(rounded(orbitals[i]; tol = actual, cap = cfg.rankcap))
        end
        # Compression must not sacrifice occupied-space orthogonality.
        defect = norm(gram(updated_orbitals; cfg) - I)
        if defect <= max(cfg.pack_gram_tolerance, 2input_defect)
            return updated_orbitals,
            (
                orbital_storage_tolerance = actual,
                orbital_rank_before_pack = maximum(rank.(orbitals)),
                stored_max_rank = maximum(rank.(updated_orbitals)),
                orbital_bytes_before_pack = before,
                orbital_pack_attempts = attempt,
                packed_gram_defect = defect,
            )
        end
    end
    error("Orbital compression failed the Gram check at requested tolerances")
end
