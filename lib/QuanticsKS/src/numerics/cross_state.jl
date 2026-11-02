# Only geometry survives a fit. Values, LU factors, and input environments are refreshed.
struct CrossState
    R::Int
    left::Vector{Vector{Vector{Int}}}
    right::Vector{Vector{Vector{Int}}}
    ranks::Vector{Int}
    errors::Vector{Float64}
    uses::Int
end
const CrossKey = Tuple{String,Int}

function remember_cross!(rec, key::CrossKey, ci, ranks, errors)
    get(rec.options, :reuse_pivots, true) || return
    isempty(first(key)) && return
    old = get(rec.cross_states, key, nothing)
    rec.cross_states[key] = CrossState(length(ci.localdims), deepcopy(ci.Iset), deepcopy(ci.Jset),
                                      copy(ranks), copy(errors), old === nothing ? 1 : old.uses + 1)
    filter!(!=(key), rec.cross_order)
    push!(rec.cross_order, key)
    while length(rec.cross_order) > get(rec.options, :pivot_cache_capacity, 512)
        delete!(rec.cross_states, popfirst!(rec.cross_order))
    end
end

function cross_state_data(rec)
    rows = [Dict("label" => key[1], "R" => s.R, "left" => s.left,
                 "right" => s.right, "ranks" => s.ranks, "errors" => s.errors, "uses" => s.uses)
            for key in rec.cross_order for s in (rec.cross_states[key],)]
    Dict("schema_version" => 1, "states" => rows)
end
function save_cross_state(rec, path)
    mkpath(dirname(abspath(path)))
    temporary = path * ".tmp"
    serialize(temporary, cross_state_data(rec))
    mv(temporary, path; force = true)
    path
end
function load_cross_state!(rec, path)
    data = deserialize(path)
    get(data, "schema_version", 0) == 1 || throw(ArgumentError("Cross-state schema"))
    staged = Pair{CrossKey,CrossState}[]
    for row in data["states"]
        R = Int(row["R"])
        1 <= R <= 60 || throw(ArgumentError("Cross-state grid"))
        left, right = row["left"], row["right"]
        length(left) == length(right) == R || throw(DimensionMismatch("Cross-state index sets"))
        for k = 1:R
            all(q -> length(q) == k - 1 && all(v -> v in (1, 2), q), left[k]) ||
                throw(ArgumentError("Cross-state left index"))
            all(q -> length(q) == R - k && all(v -> v in (1, 2), q), right[k]) ||
                throw(ArgumentError("Cross-state right index"))
            isempty(left[k]) || length(unique(left[k])) == length(left[k]) ||
                throw(ArgumentError("Duplicate cross-state index"))
            isempty(right[k]) || length(unique(right[k])) == length(right[k]) ||
                throw(ArgumentError("Duplicate cross-state index"))
            k < R && length(right[k]) != length(left[k + 1]) &&
                throw(DimensionMismatch("Cross-state bond dimensions"))
        end
        first(left) == [Int[]] && last(right) == [Int[]] ||
            throw(ArgumentError("Cross-state boundaries"))
        state = CrossState(R, deepcopy(left), deepcopy(right), Int.(row["ranks"]),
                           Float64.(row["errors"]), Int(row["uses"]))
        push!(staged, (String(row["label"]), R) => state)
    end
    for (key, state) in staged
        filter!(!=(key), rec.cross_order)
        rec.cross_states[key] = state
        push!(rec.cross_order, key)
    end
    while length(rec.cross_order) > get(rec.options, :pivot_cache_capacity, 512)
        delete!(rec.cross_states, popfirst!(rec.cross_order))
    end
    rec
end
