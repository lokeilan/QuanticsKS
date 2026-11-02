# Keep consecutive orbital transforms in coefficient space until cores are needed.

mutable struct OrbitalTransform
    basis::Vector{QTT}
    coefficients::Matrix{ComplexF64}
    metric::Matrix{ComplexF64}
    transforms::Int
    rotated::Bool
end
function OrbitalTransform(xs; metric = gram(xs))
    OrbitalTransform(collect(xs), Matrix{ComplexF64}(I, length(xs), length(xs)),
                     metric, 0, false)
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
    x.coefficients = x.coefficients *
        (fac.vectors * Diagonal(1 ./ sqrt.(fac.values)) * fac.vectors')
    x.transforms += 1
    true
end

function ritz_projection(xs, H, cfg, rec; work_tol = 0.0, potential = nothing)
    stage = work_tol > 0 ? "ritz_sampled_projection" : "ritz_direct_projection"
    timed(rec, stage) do
        work_tol > 0 || return h_gram(xs, H)
        # Apply H once per input column, at the existing work tolerance/cap.
        # The following coefficient transforms reuse this fixed-basis matrix.
        field = potential === nothing ? memoized_sample(H.potential, rec) : potential
        hx = [h_action(H, x, cfg, rec; label = "ritz_$i", work_tol, potential = field)
              for (i, x) in enumerate(xs)]
        A = gram(xs, hx)
        require_finite(A, "nonfinite_sampled_ritz_projection")
        (A + A') / 2
    end
end
function ritz_coefficients!(x::OrbitalTransform, H, cfg, rec; force = false, work_tol = 0.0, potential = nothing)
    projected = ritz_projection(x.basis, H, cfg, rec; work_tol, potential)
    A = x.coefficients' * projected * x.coefficients
    A = (A + A') / 2
    require_finite(A, "nonfinite_orbital_transform_hamiltonian")
    offdiag = norm(A - Diagonal(diag(A)))
    !force && offdiag < cfg.eigen_tol / 10 && return offdiag
    x.coefficients = x.coefficients * eigen(Hermitian(A)).vectors
    S = transform_gram(x)
    for j in axes(x.coefficients, 2)
        squared = real(S[j, j])
        isfinite(squared) && squared > 0 || error("Invalid rotated orbital norm")
        x.coefficients[:, j] ./= sqrt(squared)
    end
    x.transforms += 1
    x.rotated = true
    offdiag
end

function materialize_orbitals(x::OrbitalTransform, cfg, rec; label)
    oracle = block_oracle(x.basis, rec)
    group_size = get(rec.options, :orbital_pivot_group_size, 4)
    ys = QTT[]
    for i in axes(x.coefficients, 2)
        y = linear_combination(rec, x.basis, view(x.coefficients, :, i);
            oracle, tol = cfg.ortho_tol, cap = cfg.rankcap, label = "$(label)_$i",
            shared_pivots = "orbital_basis_group_$(cld(i, group_size))")
        push!(ys, x.rotated ? normalize(y) : y)
    end
    ys, block_cache_stats(oracle)
end

function finish_orbital_transform(x::OrbitalTransform, original, cfg, rec)
    x.transforms == 0 && return original, norm(x.metric - I)
    label = x.rotated ? "ritz_rotate" : "orthogonalize"
    ys, cache_stats = materialize_orbitals(x, cfg, rec; label)
    metric = gram(ys)
    defect = norm(metric - I)
    repaired = false
    # Compression is nonlinear: a coefficient-space identity does not certify
    # the fitted cores. Retain the original one-pass post-Ritz repair if needed.
    if x.rotated && defect >= cfg.orthogonalize_threshold
        ys, defect = orthogonalize(ys, cfg, rec; metric)
        repaired = true
    end
    event!(rec.stage_io, (; scf = rec.scf, inner = rec.inner, stage = "orbital_transform",
        parent_stage = rec.stage, seconds = 0.0, gc_seconds = 0.0, allocated_bytes = 0,
        tci_calls = 0, tci_samples = 0, tci_seconds = 0.0,
        event = "orbital_transform", basis_columns = length(x.basis),
        coefficient_transforms = x.transforms, materialization_passes = 1 + Int(repaired),
        post_fit_orthogonalization = repaired, gram_defect = defect, cache_stats...))
    ys, defect
end

function orthogonalize(xs, cfg, rec; force = false, metric = gram(xs))
    transform = OrbitalTransform(xs; metric)
    orthogonalize_coefficients!(transform, cfg; force)
    finish_orbital_transform(transform, xs, cfg, rec)
end

function ritz_rotate(xs, H, cfg, rec; work_tol = 0.0, force = false)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(xs)
    offdiag = ritz_coefficients!(transform, H, cfg, rec; force, work_tol)
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    ys, _ = finish_orbital_transform(transform, xs, cfg, rec)
    ys, offdiag
end

"Orthogonalize, optionally rotate, and rebuild only the final occupied columns."
function orthogonalize_ritz(xs, H, cfg, rec; ritz = true, work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(xs)
    orthogonalize_coefficients!(transform, cfg)
    offdiag = ritz ? ritz_coefficients!(transform, H, cfg, rec; work_tol, potential) : 0.0
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    ys, defect = finish_orbital_transform(transform, xs, cfg, rec)
    ys, defect, offdiag
end
