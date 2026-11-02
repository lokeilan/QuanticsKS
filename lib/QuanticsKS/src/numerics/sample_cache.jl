# Dense addressing on small grids; bounded CLOCK eviction on larger grids.
# A cache belongs to one fixed input function, never to a changing SCF label.
mutable struct SampleCache{T}
    values::Vector{T}
    valid::BitVector
    entries::Dict{Int,T}
    slots::Dict{Int,Int}
    keys::Vector{Int}
    recent::BitVector
    hand::Int
    capacity::Int
    evictions::Int
end

function SampleCache{T}(R; capacity = 16384, dense_limit = 0) where T
    capacity > 0 || throw(ArgumentError("Sample cache capacity must be positive"))
    n = 1 << R
    dense = n <= dense_limit ? n : 0
    SampleCache(Vector{T}(undef, dense), falses(dense), Dict{Int,T}(),
                Dict{Int,Int}(), Int[], falses(0), 1, capacity, 0)
end

function cache_get(c::SampleCache, j::Int)
    if !isempty(c.values)
        return c.valid[j + 1] ? c.values[j + 1] : nothing
    end
    slot = get(c.slots, j, 0)
    slot == 0 && return nothing
    c.recent[slot] = true
    c.entries[j]
end

function cache_put!(c::SampleCache{T}, j::Int, value) where T
    if !isempty(c.values)
        c.values[j + 1] = value
        c.valid[j + 1] = true
    elseif haskey(c.entries, j)
        c.entries[j] = value
        c.recent[c.slots[j]] = true
    else
        if length(c.keys) < c.capacity
            push!(c.keys, j)
            push!(c.recent, true)
            c.slots[j] = length(c.keys)
        else
            while c.recent[c.hand]
                c.recent[c.hand] = false
                c.hand = mod1(c.hand + 1, c.capacity)
            end
            old = c.keys[c.hand]
            delete!(c.entries, old)
            delete!(c.slots, old)
            c.keys[c.hand] = j
            c.recent[c.hand] = true
            c.slots[j] = c.hand
            c.hand = mod1(c.hand + 1, c.capacity)
            c.evictions += 1
        end
        c.entries[j] = value
    end
    value
end

Base.length(c::SampleCache) = isempty(c.values) ? length(c.entries) : count(c.valid)

function slice_indices(R, left, right, ::Val{M}) where M
    if isempty(left) || isempty(right)
        return Array{Int,M + 2}(undef, ntuple(_ -> 0, M + 2))
    end
    nl = length(first(left))
    nl + M + length(first(right)) == R || throw(DimensionMismatch("Cross slice length"))
    lp, rp = indexof.(left), indexof.(right)
    out = Array{Int}(undef, length(left), 1 << M, length(right))
    for j in eachindex(right), m = 0:((1 << M) - 1), i in eachindex(left)
        out[i, m + 1, j] = lp[i] | (m << nl) | (rp[j] << (nl + M))
    end
    reshape(out, length(left), ntuple(_ -> 2, M)..., length(right))
end

"Fill missing entries in rectangular batches without reevaluating cached points."
function cached_slice!(compute, cache::SampleCache{T}, R, left, right, middle::Val{M}) where {T,M}
    if isempty(left) || isempty(right)
        return Array{T,M + 2}(undef, ntuple(_ -> 0, M + 2)), 0, 0
    end
    indices = slice_indices(R, left, right, middle)
    result = Array{T}(undef, size(indices))
    missing = falses(size(indices))
    for k in eachindex(indices)
        value = cache_get(cache, indices[k])
        missing[k] = value === nothing
        value === nothing || (result[k] = value)
    end
    nmissing = count(missing)
    nmissing == 0 && return result, 0, 0
    if nmissing == length(result)
        values = compute(left, right, middle)
        size(values) == size(result) || throw(DimensionMismatch("Batch result shape"))
        for k in eachindex(result)
            result[k] = cache_put!(cache, indices[k], values[k])
        end
        return result, nmissing, 1
    end
    # For each central configuration, combine columns having the same missing
    # row set. Each subproblem is still a matrix contraction, with M=0.
    mask = reshape(missing, length(left), 1 << M, length(right))
    out = reshape(result, size(mask))
    ix = reshape(indices, size(mask))
    batches = 0
    for m = 1:(1 << M)
        groups = Dict{Tuple{Vararg{Int}},Vector{Int}}()
        for j in eachindex(right)
            rows = Tuple(findall(view(mask, :, m, j)))
            isempty(rows) || push!(get!(() -> Int[], groups, rows), j)
        end
        for (rows, columns) in groups
            extended = [vcat(left[i], bits(m - 1, M)) for i in rows]
            values = compute(extended, right[columns], Val(0))
            for (jj, j) in enumerate(columns), (ii, i) in enumerate(rows)
                out[i, m, j] = cache_put!(cache, ix[i, m, j], values[ii, jj])
            end
            batches += 1
        end
    end
    result, nmissing, batches
end
