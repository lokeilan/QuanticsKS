# Gram orthogonalization and Ritz rotation, rebuilding QTTs after each column transform.

function ritz_rotate(xs, H, cfg, rec; work_tol = 0.0, force = false)
    hsmall = timed(rec, "ritz_direct_projection") do
        h_gram(xs, H)
    end
    offdiag = norm(hsmall - Diagonal(diag(hsmall)))
    !force && offdiag < cfg.eigen_tol / 10 && return xs, offdiag
    C = eigen(Hermitian(hsmall)).vectors
    oracle = block_oracle(xs, rec)
    ys = [
        normalize(
            linear_combination(
                rec,
                xs,
                view(C, :, i);
                oracle,
                tol = cfg.ortho_tol,
                cap = cfg.rankcap,
                label = "ritz_rotate_$i",
            ),
        ) for i in eachindex(xs)
    ]
    first(orthogonalize(ys, cfg, rec)), offdiag
end

function orthogonalize(xs, cfg, rec; force = false)
    S = Hermitian(gram(xs))
    defect = norm(S - I)
    if defect < cfg.orthogonalize_threshold && !force
        return xs, defect
    end
    fac = eigen(S)
    minimum(fac.values) > cfg.independence_threshold ||
        error("Occupied orbitals lost linear independence")
    C = fac.vectors * Diagonal(1 ./ sqrt.(fac.values)) * fac.vectors'
    oracle = block_oracle(xs, rec)
    ys = [
        linear_combination(
            rec,
            xs,
            view(C, :, i);
            oracle,
            tol = cfg.ortho_tol,
            cap = cfg.rankcap,
            label = "orthogonalize_$i",
        ) for i in eachindex(xs)
    ]
    ys, norm(gram(ys) - I)
end
