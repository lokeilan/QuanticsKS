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

function SampleCache{T}(R; capacity = 16384, dense_limit = 0) where {T}
    capacity > 0 || throw(ArgumentError("Sample cache capacity must be positive"))
    n = 1 << R
    dense = n <= dense_limit ? n : 0
    SampleCache(
        Vector{T}(undef, dense),
        falses(dense),
        Dict{Int,T}(),
        Dict{Int,Int}(),
        Int[],
        falses(0),
        1,
        capacity,
        0,
    )
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

function cache_put!(c::SampleCache{T}, j::Int, value) where {T}
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

function map_slice(f, ::Type{T}, R, left, right, ::Val{M}) where {T,M}
    if isempty(left) || isempty(right)
        return Array{T,M + 2}(undef, ntuple(_ -> 0, M + 2))
    end
    nl = length(first(left))
    nl + M + length(first(right)) == R || throw(DimensionMismatch("Cross slice length"))
    lp = indexof.(left)
    out = Array{T}(undef, length(left), 1 << M, length(right))
    for j in eachindex(right)
        high = indexof(right[j]) << (nl + M)
        for m = 0:((1 << M) - 1), i in eachindex(left)
            out[i, m + 1, j] = f(lp[i] | (m << nl) | high)
        end
    end
    reshape(out, length(left), ntuple(_ -> 2, M)..., length(right))
end
slice_indices(R, left, right, middle::Val) = map_slice(identity, Int, R, left, right, middle)

"Fill missing entries in rectangular batches without reevaluating cached points."
function cached_slice!(compute, cache::SampleCache{T}, R, left, right, middle::Val{M}) where {T,M}
    if isempty(left) || isempty(right)
        return Array{T,M + 2}(undef, ntuple(_ -> 0, M + 2)), 0, 0
    end
    nl = length(first(left))
    nl + M + length(first(right)) == R || throw(DimensionMismatch("Cross slice length"))
    lp, rp = indexof.(left), indexof.(right)
    result = Array{T}(undef, length(left), ntuple(_ -> 2, M)..., length(right))
    missing = falses(size(result))
    out = reshape(result, length(left), 1 << M, length(right))
    mask = reshape(missing, size(out))
    for j in eachindex(right), m = 0:((1 << M) - 1), i in eachindex(left)
        value = cache_get(cache, lp[i] | (m << nl) | (rp[j] << (nl + M)))
        mask[i, m + 1, j] = value === nothing
        value === nothing || (out[i, m + 1, j] = value)
    end
    nmissing = count(missing)
    nmissing == 0 && return result, 0, 0
    if nmissing == length(result)
        values = compute(left, right, middle)
        size(values) == size(result) || throw(DimensionMismatch("Batch result shape"))
        vals = reshape(values, size(out))
        for j in eachindex(right), m = 0:((1 << M) - 1), i in eachindex(left)
            index = lp[i] | (m << nl) | (rp[j] << (nl + M))
            out[i, m + 1, j] = cache_put!(cache, index, vals[i, m + 1, j])
        end
        return result, nmissing, 1
    end
    # For each central configuration, combine columns having the same missing
    # row set. Each subproblem is still a matrix contraction, with M=0.
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
                index = lp[i] | ((m - 1) << nl) | (rp[j] << (nl + M))
                out[i, m, j] = cache_put!(cache, index, values[ii, jj])
            end
            batches += 1
        end
    end
    result, nmissing, batches
end

# Use endpoint dot products when both ends cover the grid; contract larger-grid slices.
abstract type SampledFunction <: TCI.BatchEvaluator{ComplexF64} end
(f::SampledFunction)(q::AbstractVector{Int}) = f(indexof(q))
native_batch(::SampledFunction) = false
scalar_fastpath(::SampledFunction) = false
sampling_method(f::SampledFunction) =
    scalar_fastpath(f) ? "cached_scalar" : native_batch(f) ? "tt_slices" : "coordinate"

struct CoordinateSamples{F} <: SampledFunction
    f::F
    R::Int
end
(f::CoordinateSamples)(j::Int) = ComplexF64(f.f(j))
sampled(f, R::Int) = CoordinateSamples(f, R)
sampled(f::SampledFunction, R::Int) =
    sample_length(f) == R ? f : throw(DimensionMismatch("Sample grid"))
sample_length(f::CoordinateSamples) = f.R
scalar_fastpath(::CoordinateSamples) = true
function (f::CoordinateSamples)(left, right, middle::Val)
    map_slice(f, ComplexF64, f.R, left, right, middle)
end

struct TensorSamples <: SampledFunction
    evaluator::Evaluator
    frames::Union{Nothing,TCI.TTCache{ComplexF64}}
    frame_capacity::Int
end
function TensorSamples(e::Evaluator; frame_capacity = 2048)
    frames = e.prefix + e.suffix == length(e.x) ? nothing : TCI.TTCache(e.x.c)
    TensorSamples(e, frames, frame_capacity)
end
sampled(x::QTT, rec) =
    TensorSamples(evaluate(x, rec); frame_capacity = get(rec.options, :slice_cache_capacity, 2048))
sample_length(f::TensorSamples) = length(f.evaluator.x)
native_batch(::TensorSamples) = true
scalar_fastpath(f::TensorSamples) = f.evaluator.prefix + f.evaluator.suffix == sample_length(f)
(f::TensorSamples)(j::Int) = f.evaluator(j)

function trim_frames!(frames, capacity)
    caches = (frames.cacheleft..., frames.cacheright...)
    remaining = sum(length, caches)
    while remaining > capacity
        for cache in caches
            if !isempty(cache)
                delete!(cache, first(keys(cache)))
                remaining -= 1
                remaining <= capacity && break
            end
        end
    end
end
function (f::TensorSamples)(left, right, middle::Val)
    frames = f.frames
    frames === nothing &&
        return map_slice(f.evaluator, ComplexF64, sample_length(f), left, right, middle)
    result = TCI.batchevaluate(frames, left, right, middle)
    trim_frames!(frames, f.frame_capacity)
    result
end

struct PointwiseSamples{F,A} <: SampledFunction
    f::F
    arguments::A
end
function pointwise(f, args::SampledFunction...)
    isempty(args) && throw(ArgumentError("A sampled expression needs operands"))
    all(a -> sample_length(a) == sample_length(first(args)), args) ||
        throw(DimensionMismatch("Sampled expression grids"))
    PointwiseSamples(f, args)
end
sample_length(f::PointwiseSamples) = sample_length(first(f.arguments))
native_batch(f::PointwiseSamples) = any(native_batch, f.arguments)
scalar_fastpath(f::PointwiseSamples) = all(scalar_fastpath, f.arguments)
(f::PointwiseSamples)(j::Int) = ComplexF64(f.f(map(a -> a(j), f.arguments)...))
function (f::PointwiseSamples)(left, right, middle::Val)
    values = map(a -> a(left, right, middle), f.arguments)
    ComplexF64.(broadcast(f.f, values...))
end

struct ShiftedSamples{F} <: SampledFunction
    input::F
    offset::Int
end
shifted(f::SampledFunction, offset::Int) = ShiftedSamples(f, offset)
sample_length(f::ShiftedSamples) = sample_length(f.input)
native_batch(f::ShiftedSamples) = native_batch(f.input)
scalar_fastpath(f::ShiftedSamples) = scalar_fastpath(f.input)
(f::ShiftedSamples)(j::Int) = f.input(mod(j + f.offset, 1 << sample_length(f)))
function (f::ShiftedSamples)(left, right, ::Val{M}) where {M}
    if isempty(left) || isempty(right)
        return Array{ComplexF64,M + 2}(undef, ntuple(_ -> 0, M + 2))
    end
    nl, nr = length(first(left)), length(first(right))
    out = Array{ComplexF64}(undef, length(left), 1 << M, length(right))
    # A periodic shift only couples the three index groups through a carry.
    # Partition by that carry and retain a rectangular matrix contraction.
    groups = Dict{Int,Vector{Int}}()
    for i in eachindex(left)
        carry = fld(indexof(left[i]) + f.offset, 1 << nl)
        push!(get!(() -> Int[], groups, carry), i)
    end
    for (carry, rows) in groups, m = 0:((1 << M) - 1)
        rightcarry, center = fldmod(m + carry, 1 << M)
        l = [
            vcat(bits(mod(indexof(left[i]) + f.offset, 1 << nl), nl), bits(center, M)) for i in rows
        ]
        r = [bits(mod(indexof(q) + rightcarry, 1 << nr), nr) for q in right]
        values = f.input(l, r, Val(0))
        for j in eachindex(right), (ii, i) in enumerate(rows)
            out[i, m + 1, j] = values[ii, j]
        end
    end
    reshape(out, length(left), ntuple(_ -> 2, M)..., length(right))
end

mutable struct RecordedSamples{F} <: SampledFunction
    input::F
    cache::SampleCache{ComplexF64}
    R::Int
    requests::Int
    evaluations::Int
    batches::Int
    computed_batches::Int
    batch_evaluations::Int
end
function RecordedSamples(f::SampledFunction, options)
    R = sample_length(f)
    cache = SampleCache{ComplexF64}(
        R;
        capacity = options.scalar_cache_capacity,
        dense_limit = get(options, :dense_cache_max_points, 1 << 18),
    )
    RecordedSamples(f, cache, R, 0, 0, 0, 0, 0)
end
sample_length(f::RecordedSamples) = f.R
native_batch(f::RecordedSamples) = native_batch(f.input)
scalar_fastpath(f::RecordedSamples) = scalar_fastpath(f.input)
function (f::RecordedSamples)(j::Int)
    0 <= j < (1 << f.R) || throw(BoundsError(f, j))
    f.requests += 1
    value = cache_get(f.cache, j)
    if value === nothing
        value = ComplexF64(f.input(j))
        isfinite(value) || throw(ArgumentError("Nonfinite sampled value at index $j"))
        cache_put!(f.cache, j, value)
        f.evaluations += 1
    end
    value
end
function (f::RecordedSamples)(left, right, middle::Val{M}) where {M}
    f.batches += 1
    if scalar_fastpath(f)
        # Endpoint dot products are cheaper than grouping sparse misses and
        # building TTCache environments for every cross cut on small grids.
        before = f.evaluations
        result = map_slice(f, ComplexF64, f.R, left, right, middle)
        evaluated = f.evaluations - before
        f.batch_evaluations += evaluated
        f.computed_batches += Int(evaluated > 0)
        return result
    end
    f.requests += length(left) * (1 << M) * length(right)
    result, evaluations, batches = cached_slice!(f.cache, f.R, left, right, middle) do l, r, m
        values = f.input(l, r, m)
        all(isfinite, values) || throw(ArgumentError("Nonfinite sampled cross slice"))
        values
    end
    f.evaluations += evaluations
    f.computed_batches += batches
    f.batch_evaluations += evaluations
    result
end
memoized_sample(x::QTT, rec) = RecordedSamples(sampled(x, rec), rec.options)
memoized_sample(x::SampledFunction, rec) = RecordedSamples(x, rec.options)

uses_tt_frames(::SampledFunction) = false
uses_tt_frames(f::TensorSamples) = f.frames !== nothing
uses_tt_frames(f::PointwiseSamples) = any(uses_tt_frames, f.arguments)
uses_tt_frames(f::ShiftedSamples) = uses_tt_frames(f.input)
uses_tt_frames(f::RecordedSamples) = uses_tt_frames(f.input)

# The warm-start scale must see the current function on every old slice before
# optimize! starts. Stream endpoint values without allocating a result tensor.
function maximum_slice(f::RecordedSamples, left, right, middle::Val{M}) where {M}
    scalar_fastpath(f) || return maximum(abs, f(left, right, middle); init = 0.0)
    f.batches += 1
    before = f.evaluations
    result = 0.0
    if !isempty(left) && !isempty(right)
        nl = length(first(left))
        nl + M + length(first(right)) == f.R || throw(DimensionMismatch("Cross slice length"))
        lp = indexof.(left)
        for q in right
            high = indexof(q) << (nl + M)
            for m = 0:((1 << M) - 1), low in lp
                result = max(result, abs(f(low | (m << nl) | high)))
            end
        end
    end
    evaluated = f.evaluations - before
    f.batch_evaluations += evaluated
    f.computed_batches += Int(evaluated > 0)
    result
end

mutable struct BlockOracle
    evals::Vector{Evaluator}
    samples::Vector{TensorSamples}
    cache::SampleCache{Vector{ComplexF64}}
    cap::Int
    requests::Int
    evaluations::Int
    endpoint_fastpath::Bool
    parallel_tasks::Int
end
function BlockOracle(
    xs;
    cap = 8192,
    cachebits = 10,
    frame_capacity = 2048,
    dense_limit = 0,
    max_bytes = 64 * 1024^2,
    parallel_tasks = 1,
)
    evals = [Evaluator(x; cachebits) for x in xs]
    samples = [TensorSamples(e; frame_capacity) for e in evals]
    R = length(first(xs))
    # Include vector headers and references; values are populated only on demand.
    bytes_per_point = sizeof(ComplexF64) * length(evals) + 64
    limit = min(dense_limit, max_bytes ÷ bytes_per_point)
    cache = SampleCache{Vector{ComplexF64}}(R; capacity = cap, dense_limit = limit)
    BlockOracle(
        evals,
        samples,
        cache,
        isempty(cache.values) ? cap : length(cache.values),
        0,
        0,
        all(scalar_fastpath, samples),
        parallel_tasks,
    )
end
scalar_fastpath(o::BlockOracle) = o.endpoint_fastpath
function (o::BlockOracle)(j::Int)
    o.requests += 1
    value = cache_get(o.cache, j)
    if value === nothing
        value = cache_put!(o.cache, j, [e(j) for e in o.evals])
        o.evaluations += 1
    end
    value
end
function (o::BlockOracle)(left, right, middle::Val{M}) where {M}
    R = sample_length(first(o.samples))
    scalar_fastpath(o) && return map_slice(o, Vector{ComplexF64}, R, left, right, middle)
    o.requests += length(left) * (1 << M) * length(right)
    result, evaluations, _ = cached_slice!(o.cache, R, left, right, middle) do l, r, m
        columns = independent_map(
            Array{ComplexF64,M + 2},
            length(o.samples),
            (; parallel_tasks = o.parallel_tasks),
        ) do i
            o.samples[i](l, r, m)
        end
        out = Array{Vector{ComplexF64}}(undef, size(first(columns)))
        for i in eachindex(out)
            out[i] = ComplexF64[c[i] for c in columns]
        end
        out
    end
    o.evaluations += evaluations
    result
end
block_cache_stats(o::BlockOracle) = (;
    block_cache_mode = isempty(o.cache.values) ? "bounded_clock" : "dense_addressed",
    block_cache_capacity = o.cap,
    block_cache_evictions = o.cache.evictions,
    block_sample_requests = o.requests,
    block_sample_evaluations = o.evaluations,
)

struct LinearSamples{O} <: SampledFunction
    oracle::O
    weights::Vector{ComplexF64}
end
sample_length(f::LinearSamples) = sample_length(f.oracle)
native_batch(::LinearSamples) = true
scalar_fastpath(f::LinearSamples) = scalar_fastpath(f.oracle)
# Linear-combination coefficients are not conjugated.
(f::LinearSamples)(j::Int) = transpose(f.weights) * f.oracle(j)
function (f::LinearSamples)(left, right, middle::Val)
    values = f.oracle(left, right, middle)
    isempty(values) && return zeros(ComplexF64, size(values))
    map(v -> transpose(f.weights) * v, values)
end

struct DensitySamples{O} <: SampledFunction
    oracle::O
    weights::Vector{Float64}
end
sample_length(f::DensitySamples) = sample_length(f.oracle)
native_batch(::DensitySamples) = true
scalar_fastpath(f::DensitySamples) = scalar_fastpath(f.oracle)
weighted_density(weights, values) =
    ComplexF64(sum(weights[i] * abs2(values[i]) for i in eachindex(weights, values)))
(f::DensitySamples)(j::Int) = weighted_density(f.weights, f.oracle(j))
function (f::DensitySamples)(left, right, middle::Val)
    map(v -> weighted_density(f.weights, v), f.oracle(left, right, middle))
end

sample_length(o::BlockOracle) = sample_length(first(o.samples))

struct SynchronizedSamples{F<:SampledFunction} <: SampledFunction
    input::F
    mutex::ReentrantLock
end
struct SynchronizedBlockOracle
    input::BlockOracle
    mutex::ReentrantLock
end
synchronized(f::SampledFunction) = SynchronizedSamples(f, ReentrantLock())
synchronized(f::BlockOracle) = SynchronizedBlockOracle(f, ReentrantLock())
synchronized(f::Union{SynchronizedSamples,SynchronizedBlockOracle}) = f
sample_length(f::Union{SynchronizedSamples,SynchronizedBlockOracle}) = sample_length(f.input)
native_batch(f::SynchronizedSamples) = native_batch(f.input)
scalar_fastpath(f::Union{SynchronizedSamples,SynchronizedBlockOracle}) = scalar_fastpath(f.input)
sampling_method(f::SynchronizedSamples) = sampling_method(f.input)
uses_tt_frames(f::SynchronizedSamples) = uses_tt_frames(f.input)
function (f::Union{SynchronizedSamples,SynchronizedBlockOracle})(j::Int)
    lock(f.mutex) do
        f.input(j)
    end
end
function (f::Union{SynchronizedSamples,SynchronizedBlockOracle})(left, right, middle::Val)
    lock(f.mutex) do
        f.input(left, right, middle)
    end
end
block_cache_stats(f::SynchronizedBlockOracle) = block_cache_stats(f.input)
