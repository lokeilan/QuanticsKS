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

function h_evaluator(H, x, rec; potential = nothing)
    ev = potential === nothing ? potential_samples(H, rec) : potential
    HamiltonianEvaluator(evaluate(x, rec), ev, 2^length(x), 1 / (2H.h^2))
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
function h_core_fallback(H, x, energy; tol, cap)
    shift_tol = min(tol, tol * H.h^2)
    terms = (
        x,
        apply_mpo(H.diagonal, x; tol, cap),
        apply_mpo(H.plus, x; tol = shift_tol, cap),
        apply_mpo(H.minus, x; tol = shift_tol, cap),
    )
    weights = (1 / H.h^2 - energy, 1, -1 / (2H.h^2), -1 / (2H.h^2))
    materialize_expression(linear_expression(terms, weights); tol, cap)
end

"Sample just the final FD2 residual, without fitting H*x, V*x or shifted states."
function h_residual(
    H,
    x,
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
    orbital = sampled(x, rec)
    field =
        potential === nothing ? potential_samples(H, rec) :
        potential isa SampledFunction ? potential :
        potential isa Evaluator ?
        TensorSamples(potential; frame_capacity = get(rec.options, :slice_cache_capacity, 2048)) :
        sampled(potential, length(x))
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
        length(x),
        fn;
        tol,
        cap,
        label,
        initial_zero_fallback = () -> h_core_fallback(H, x, energy; tol, cap),
    )
end

function h_action(H, x, cfg, rec; label = "", work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid H-action tolerance"))
    tol = work_tol > 0 ? work_tol : cfg.round_tol
    h_residual(
        H,
        x,
        0.0,
        cfg,
        rec;
        tol,
        potential,
        label = (work_tol > 0 ? "Hpsi_native_" : "Hpsi_") * label,
    )
end
function h_project(f, x, H)
    project(f, x, H.diagonal) +
    (project(f, x) - (project(f, x, H.plus) + project(f, x, H.minus)) / 2) / H.h^2
end

"Project H on gauge-constrained tangent coordinates, such as L4's projected/SVD columns."
function h_project_tangent(f, z, H)
    t = tangent_tt(f, z)
    project(f, t, H.diagonal) + (z - (project(f, t, H.plus) + project(f, t, H.minus)) / 2) / H.h^2
end
function h_expectation(H, x)
    # All coefficient-Hamiltonian blocks must use the same MPO contraction.
    # A loose TCI Rayleigh diagonal becomes an artificial negative direction
    # when the joint old-orbital / tangent Gram matrix is nearly singular.
    real(only(h_gram([x], H)))
end

function h_projection_gradient(H, x)
    # Exact low-rank shift, used only for cancellation-prone kinetic entries.
    # No sampled fit or finite-tolerance truncation enters this fallback.
    cap = maximum(size(w, 4) * size(g, 3) for (w, g) in zip(H.plus, x.c))
    shifted = apply_mpo(H.plus, x; tol = 0.0, cap)
    materialize_expression(
        linear_expression((shifted, x), (1, -1));
        tol = 0.0,
        cap = rank(shifted) + rank(x),
    )
end

"Project H from cores; use stable gradient contractions when FD2 terms cancel."
function h_gram(xs, H; cfg = (;))
    parallel_workers(cfg, length(xs)) > 1 && return parallel_h_gram(xs, H, cfg)
    A = zeros(ComplexF64, length(xs), length(xs))
    gradients = Dict{Int,QTT}()
    gradient(i) = get!(() -> h_projection_gradient(H, xs[i]), gradients, i)
    for j in eachindex(xs), i = 1:j
        terms = (
            inner(xs[i], xs[j]; cfg),
            -inner_mpo(xs[i], H.plus, xs[j]) / 2,
            -inner_mpo(xs[i], H.minus, xs[j]) / 2,
        )
        numerator = compensated_sum(terms)
        cancellation =
            abs(numerator) <= sqrt(eps(Float64)) * sum(abs, terms) ||
            (i == j && real(numerator) < 0)
        kinetic = cancellation ? inner(gradient(i), gradient(j); cfg) / (2H.h^2) : numerator / H.h^2
        value = inner_mpo(xs[i], H.diagonal, xs[j]) + kinetic
        A[i, j] = value
        A[j, i] = conj(value)
        i == j && (A[i, i] = real(value))
    end
    require_finite(A, "nonfinite_direct_hamiltonian_gram")
    A
end

function parallel_h_gram(xs, H, cfg)
    b = length(xs)
    A = zeros(ComplexF64, b, b)
    fallback = fill(false, b, b)
    foreach_independent(b, cfg) do j
        for i = 1:j
            terms = (
                inner(xs[i], xs[j]; cfg),
                -inner_mpo(xs[i], H.plus, xs[j]) / 2,
                -inner_mpo(xs[i], H.minus, xs[j]) / 2,
            )
            numerator = compensated_sum(terms)
            fallback[i, j] =
                abs(numerator) <= sqrt(eps(Float64)) * sum(abs, terms) ||
                (i == j && real(numerator) < 0)
            A[i, j] = inner_mpo(xs[i], H.diagonal, xs[j]) + (fallback[i, j] ? 0 : numerator / H.h^2)
        end
    end
    needed = [i for i = 1:b if any(fallback[min(i, j), max(i, j)] for j = 1:b)]
    gradients = Vector{QTT}(undef, b)
    foreach_independent(length(needed), cfg) do k
        i = needed[k]
        gradients[i] = h_projection_gradient(H, xs[i])
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
