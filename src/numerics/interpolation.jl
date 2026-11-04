# Interpolate neighboring periodic samples to evaluate a QTT on a finer grid.

"Periodic degree-five interpolation of normalized grid amplitudes, using six TT point queries."
struct PeriodicRefinement
    evaluate::Evaluator
    coarse_size::Int
    ratio::Int
    nodes::Vector{Int}
    weights::Matrix{Float64}
end
function PeriodicRefinement(x::QTT, target_R::Int; degree = 5, cachebits = 10)
    target_R > length(x) || throw(ArgumentError("Refinement requires a finer grid"))
    q = 1 << (target_R - length(x))
    nodes = collect((-(degree ÷ 2)):(degree ÷ 2 + 1))
    weights = [
        prod((s / q - b) / (a - b) for b in nodes if b != a) / sqrt(q) for a in nodes, s = 0:(q - 1)
    ]
    PeriodicRefinement(Evaluator(x; cachebits), 1 << length(x), q, nodes, weights)
end
function (p::PeriodicRefinement)(j::Int)
    coarse, offset = divrem(j, p.ratio)
    offset == 0 && return p.evaluate(mod(coarse, p.coarse_size)) / sqrt(p.ratio)
    sum(
        p.weights[k, offset + 1] * p.evaluate(mod(coarse + p.nodes[k], p.coarse_size)) for
        k in eachindex(p.nodes)
    )
end
