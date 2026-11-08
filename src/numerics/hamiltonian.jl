# Project and rebuild tangent vectors in one orthogonal frame with shared core offsets.

struct Frame
    left::Vector{Array{ComplexF64,3}}
    right::Vector{Array{ComplexF64,3}}
    shapes::Vector{Tuple{Int,Int,Int}}
    offsets::Vector{Int}
end
function canonical_frame(x::QTT; cfg = (;))
    # Tangent bases need orthogonality, not another rank truncation.
    u, v = with_sampling_seed(0) do
        psi = ITensorMPS.MPS(tensortrain(x))
        ITensorMPS.orthogonalize!(psi, 1)
        right = QTT(psi).c
        ITensorMPS.orthogonalize!(psi, length(x))
        QTT(psi).c, right
    end
    d = length(x)
    sh = [(size(u[k], 1), 2, size(v[k], 3)) for k = 1:d]
    off = cumsum(vcat(0, [prod(s) for s in sh]))
    Frame(u, v, sh, off)
end
function pack_delta(f::Frame, gs)
    reduce(vcat, vec.(gs))
end
function unpack_delta(f::Frame, z)
    [reshape(view(z, (f.offsets[k] + 1):f.offsets[k + 1]), f.shapes[k]) for k = 1:length(f.shapes)]
end
# Remove the component parallel to the left basis to avoid redundant tangent coordinates.
function gauge!(g, u)
    m = reshape(g, :, size(g, 3))
    q = reshape(u, :, size(u, 3))
    m .-= q * (q' * m)
    g
end
"Project an MPO action directly into tangent coordinates, without constructing W*x."
function project(f::Frame, x::QTT, w = nothing)
    d = length(x)
    if w === nothing
        w = [reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1) for k = 1:d]
    end
    right = Vector{Array{ComplexF64,3}}(undef, d + 1)
    right[d + 1] = ones(ComplexF64, 1, 1, 1)
    for k = d:-1:1
        v = f.right[k]
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        @tensor t1[l, s, I, J] := z[l, s, L] * t[I, J, L]
        @tensor t2[j, s, l, I] := a[j, s, t, J] * t1[l, t, I, J]
        @tensor e[i, j, l] := conj(v[i, s, I]) * t2[j, s, l, I]
        right[k] = e
    end
    left = ones(ComplexF64, 1, 1, 1)
    ds = Array{ComplexF64,3}[]
    for k = 1:d
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        u = f.left[k]
        @tensor t1[i, j, s, L] := left[i, j, l] * z[l, s, L]
        @tensor t2[i, s, J, L] := t1[i, j, t, L] * a[j, s, t, J]
        @tensor g[i, s, I] := t2[i, s, J, L] * t[I, J, L]
        k < d && gauge!(g, u)
        push!(ds, g)
        if k < d
            @tensor e[I, J, L] := conj(u[i, s, I]) * t2[i, s, J, L]
            left = e
        end
    end
    pack_delta(f, ds)
end
function tangent_tt(f::Frame, z)
    ds = unpack_delta(f, z)
    c = Array{ComplexF64,3}[]
    d = length(ds)
    for k = 1:d
        u = f.left[k]
        v = f.right[k]
        a = ds[k]
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
struct HamiltonianEvaluator{X,V}
    orbital::X
    potential::V
    N::Int
    inverse_two_h_squared::Float64
end

function h_evaluator(H, psi, rec; potential = nothing)
    ev = potential === nothing ? potential_samples(H, rec) : potential
    HamiltonianEvaluator(evaluate(psi, rec), ev, 2^length(psi), 1 / (2H.h^2))
end

# Take neighboring differences before adding the local potential/residual term.
function residual_value(e::HamiltonianEvaluator, j::Int, energy)
    center = e.orbital(j)
    gradient = (center - e.orbital(mod(j - 1, e.N))) + (center - e.orbital(mod(j + 1, e.N)))
    gradient * e.inverse_two_h_squared + (e.potential(j) - energy) * center
end
(e::HamiltonianEvaluator)(j::Int) = residual_value(e, j, 0)

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
    materialize_expression(linear_expression(terms, weights); tol, cap)
end

"Sample just the final FD2 residual, without fitting H*x, V*x or shifted states."
function h_residual(
    H,
    psi,
    energy,
    cfg,
    rec;
    tol = cfg.round_tol,
    cap = cfg.workcap,
    label = "orbital_residual",
    potential = nothing,
)
    algebra_options(tol, cap)
    isfinite(energy) || throw(ArgumentError("Nonfinite orbital energy"))
    orbital = sampled(psi, rec)
    field =
        potential === nothing ? potential_samples(H, rec) :
        potential isa SampledFunction ? potential :
        potential isa Evaluator ?
        TensorSamples(potential; frame_capacity = get(rec.options, :slice_cache_capacity, 2048)) :
        sampled(potential, length(psi))
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
    tol = work_tol > 0 ? work_tol : cfg.round_tol
    h_residual(
        H,
        psi,
        0.0,
        cfg,
        rec;
        tol,
        potential,
        label = (work_tol > 0 ? "Hpsi_native_" : "Hpsi_") * label,
    )
end
function h_project(f, psi, H)
    project(f, psi, H.diagonal) +
    (project(f, psi) - (project(f, psi, H.plus) + project(f, psi, H.minus)) / 2) / H.h^2
end

"Project H on gauge-constrained tangent coordinates, such as L4's projected/SVD columns."
function h_project_tangent(f, z, H)
    t = tangent_tt(f, z)
    project(f, t, H.diagonal) + (z - (project(f, t, H.plus) + project(f, t, H.minus)) / 2) / H.h^2
end
function h_expectation(H, psi)
    # All coefficient-Hamiltonian blocks must use the same MPO contraction.
    # A loose TCI Rayleigh diagonal becomes an artificial negative direction
    # when the joint old-orbital / tangent Gram matrix is nearly singular.
    real(only(h_gram([psi], H)))
end

function h_projection_gradient(H, psi)
    # Exact low-rank shift, used only for cancellation-prone kinetic entries.
    # No sampled fit or finite-tolerance truncation enters this fallback.
    cap = maximum(size(w, 4) * size(g, 3) for (w, g) in zip(H.plus, psi.c))
    shifted = apply_mpo(H.plus, psi; tol = 0.0, cap)
    materialize_expression(
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
            gradients[i] = h_projection_gradient(H, orbitals[i])
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
