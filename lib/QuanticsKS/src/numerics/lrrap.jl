# Low-rank orbital update; stages L1 to L7 match the performance logs.

include("hamiltonian.jl")
include("orbital_basis.jl")
include("coefficients.jl")

"tensorchem-style Block-Green correction; no differentiated TCI error in the RHS."
function block_green(xs, energies, H, green, cfg, rec)
    ev = memoized_sample(H.potential, rec)
    ys = QTT[]
    for (i, x) in enumerate(xs)
        ex = sampled(x, rec)
        rhs = cross(
            rec,
            cfg.R,
            pointwise((v, value) -> (cfg.green_shift + energies[i] - v) * value, ev, ex);
            tol = cfg.rhs_tol,
            cap = get(cfg, :rhs_cap, cfg.workcap),
            label = "block_green_$i",
        )
        push!(ys, normalize(apply_mpo(green, rhs; tol = cfg.round_tol, cap = cfg.rankcap)))
    end
    ys
end
function lrrap_step(xs, previous, H, green, cfg, rec; work_tol = 0.0)
    isfinite(work_tol) && work_tol >= 0 ||
        throw(ArgumentError("work_tol must be finite and nonnegative"))
    algebra_tol = work_tol > 0 ? work_tol : cfg.round_tol
    state_tol = min(algebra_tol, cfg.tci_tol)
    b = length(xs)
    # L1: contract Rayleigh energies and sample only the final residuals.
    eps, residuals = timed(rec, "L1") do
        eps = Float64[]
        rs = QTT[]
        ev = memoized_sample(H.potential, rec)
        for i = 1:b
            e = h_expectation(H, xs[i]) / real(inner(xs[i], xs[i]))
            push!(eps, e)
            push!(
                rs,
                h_residual(
                    H, xs[i], e, cfg, rec;
                    tol = algebra_tol,
                    cap = cfg.workcap,
                    label = "orbital_residual_$i",
                    potential = ev,
                ),
            )
        end
        eps, rs
    end
    norms = ttnorm.(residuals)
    # Sweep the tangent base over the occupied set; a converged low state must not monopolize it.
    anchor = mod((rec.scf - 1) * cfg.inner_steps + rec.inner - 1, b) + 1
    frame = timed(rec, "L2") do
        canonical_frame(xs[anchor])
    end
    # L3: project orbitals, Green-preconditioned residuals, and prior directions into one tangent space.
    DX, DZ, DP = timed(rec, "L3") do
        dx = hcat([project(frame, x) for x in xs]...)
        dz = hcat(
            [
                project(frame, r, green) for r in residuals
            ]...,
        )
        dp =
            previous === nothing ? zeros(ComplexF64, size(dx, 1), 0) :
            hcat([project(frame, p) for p in previous]...)
        dx, dz, dp
    end
    # Only tangent coordinates are needed beyond this point.
    empty!(residuals)
    D, W, HV, Vx, Vhx, Sxx = timed(rec, "L4") do
        raw = hcat(DX, DZ, DP)
        # Whiten only the shared tangent coordinates. Old X retains diagonal coefficients.
        column_norms = [max(norm(view(raw, :, j)), 1e-14) for j = 1:size(raw, 2)]
        for j = 1:size(raw, 2)
            view(raw, :, j) ./= column_norms[j]
        end
        # Overwrite the normalized coordinate buffer and retain a view of U.
        # This avoids three extra copies of the largest tangent-coordinate array.
        fac = svd!(raw; alg = LinearAlgebra.QRIteration())
        good = findall(
            >(max(maximum(fac.S) * cfg.tangent_relative_cutoff, cfg.tangent_absolute_cutoff)),
            fac.S,
        )
        W = Diagonal(1 ./ column_norms) * fac.V[:, good] * Diagonal(1 ./ fac.S[good])
        D = view(fac.U, :, 1:length(good))
        HV = zeros(ComplexF64, size(D, 2), size(D, 2))
        for j = 1:size(D, 2)
            hv = h_project(frame, tangent_tt(frame, view(D, :, j)), H)
            HV[:, j] = D' * hv
            if j % 20 == 0
                println("    L4 Hamiltonian tangent contraction $j/$(size(D,2))")
                flush(stdout)
            end
        end
        HV = (HV + HV') / 2
        # Exact contractions with H, independently of the TCI approximation used in L1.
        phx = hcat([h_project(frame, x, H) for x in xs]...)
        D, W, HV, D' * DX, D' * phx, gram(xs)
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
            nsweeps = cfg.coefficient_sweeps,
            metric_relative_cutoff = cfg.metric_relative_cutoff,
            metric_absolute_cutoff = cfg.metric_absolute_cutoff,
            constraint_relative_cutoff = cfg.constraint_relative_cutoff,
            constraint_absolute_cutoff = cfg.constraint_absolute_cutoff,
        )
    end
    # L6: rebuild orbitals and the search directions for the next iteration.
    ys, ps = timed(rec, "L6") do
        rawC = W * co.C
        corr = D * co.C
        pcorr = DZ * rawC[(b + 1):(2b), :]
        previous !== nothing && (pcorr .+= DP * rawC[(2b + 1):end, :])
        ys = QTT[]
        ps = QTT[]
        for i = 1:b
            t = tangent_tt(frame, view(corr, :, i))
            push!(
                ys,
                normalize(
                    add(
                        xs[i],
                        t;
                        a = co.c[i],
                        tol = state_tol,
                        cap = cfg.rankcap,
                        rec,
                        label = "orbital_update_$i",
                    ),
                ),
            )
            push!(
                ps,
                with_svd_context(;
                    orbital = i,
                    role = "history_direction",
                    coordinates = view(pcorr, :, i),
                ) do
                    rounded(tangent_tt(frame, view(pcorr, :, i)); tol = algebra_tol, cap = cfg.rankcap)
                end,
            )
        end
        ys, ps
    end
    polish =
        (cfg.green_polish_every > 0 && rec.scf % cfg.green_polish_every == 0) ||
        maximum(norms) < cfg.green_polish_residual
    if polish
        # Accepted orbitals keep the original tight Green output accuracy;
        # the looser work budget applies to search directions, not this state.
        ys = timed(rec, "L6G") do
            block_green(ys, eps, H, green, cfg, rec)
        end
        ps = [constant_tt(cfg.R, 0.0) for i = 1:b]
    end
    ys, gramdef, ritz_offdiag = timed(rec, "L7") do
        orthogonalize_ritz(ys, H, cfg, rec; ritz = polish, work_tol)
    end
    metrics = (
        anchor = anchor,
        tangent_dimension = size(D, 2),
        residual_max = maximum(norms),
        residual_rms = sqrt(sum(abs2, norms) / b),
        trace_before = sum(eps),
        trace_after = co.trace_after,
        coefficient_gram_defect = co.gram_defect,
        gram_defect = gramdef,
        green_polish = polish,
        ritz_offdiagonal_before = ritz_offdiag,
        max_rank = maximum(rank.(ys)),
        orbital_bytes = sum(storage, ys),
        history_bytes = sum(storage, ps),
        work_tolerance = work_tol,
        work_rank_cap = cfg.workcap,
    )
    ys, ps, metrics
end
