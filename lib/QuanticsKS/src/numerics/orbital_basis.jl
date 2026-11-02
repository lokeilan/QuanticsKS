# Keep consecutive orbital transforms in coefficient space until cores are needed.

mutable struct OrbitalTransform
    basis::Vector{QTT}
    coefficients::Matrix{ComplexF64}
    metric::Matrix{ComplexF64}
    transforms::Int
    rotated::Bool
end
function OrbitalTransform(xs)
    OrbitalTransform(collect(xs), Matrix{ComplexF64}(I, length(xs), length(xs)),
                     gram(xs), 0, false)
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

function ritz_coefficients!(x::OrbitalTransform, H, cfg, rec; force = false)
    # H is fixed during this sequence. Contract its projected matrix once;
    # coefficient transforms never require sampled H*psi intermediates.
    projected = timed(rec, "ritz_direct_projection") do
        h_gram(x.basis, H)
    end
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
    ys
end

function finish_orbital_transform(x::OrbitalTransform, original, cfg, rec)
    x.transforms == 0 && return original, norm(x.metric - I)
    label = x.rotated ? "ritz_rotate" : "orthogonalize"
    ys = materialize_orbitals(x, cfg, rec; label)
    defect = norm(gram(ys) - I)
    repaired = false
    # Compression is nonlinear: a coefficient-space identity does not certify
    # the fitted cores. Retain the original one-pass post-Ritz repair if needed.
    if x.rotated && defect >= cfg.orthogonalize_threshold
        ys, defect = orthogonalize(ys, cfg, rec)
        repaired = true
    end
    event!(rec.stage_io, (; scf = rec.scf, inner = rec.inner, stage = "orbital_transform",
        parent_stage = rec.stage, seconds = 0.0, gc_seconds = 0.0, allocated_bytes = 0,
        tci_calls = 0, tci_samples = 0, tci_seconds = 0.0,
        event = "orbital_transform", basis_columns = length(x.basis),
        coefficient_transforms = x.transforms, materialization_passes = 1 + Int(repaired),
        post_fit_orthogonalization = repaired, gram_defect = defect))
    ys, defect
end

function orthogonalize(xs, cfg, rec; force = false)
    transform = OrbitalTransform(xs)
    orthogonalize_coefficients!(transform, cfg; force)
    finish_orbital_transform(transform, xs, cfg, rec)
end

function ritz_rotate(xs, H, cfg, rec; work_tol = 0.0, force = false)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(xs)
    offdiag = ritz_coefficients!(transform, H, cfg, rec; force)
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    ys, _ = finish_orbital_transform(transform, xs, cfg, rec)
    ys, offdiag
end

"Orthogonalize, optionally rotate, and rebuild only the final occupied columns."
function orthogonalize_ritz(xs, H, cfg, rec; ritz = true, work_tol = 0.0)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(xs)
    orthogonalize_coefficients!(transform, cfg)
    offdiag = ritz ? ritz_coefficients!(transform, H, cfg, rec) : 0.0
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    ys, defect = finish_orbital_transform(transform, xs, cfg, rec)
    ys, defect, offdiag
end
