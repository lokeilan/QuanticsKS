# Use endpoint dot products when both ends cover the grid; contract larger-grid slices.
abstract type SampledFunction <: TCI.BatchEvaluator{ComplexF64} end
(f::SampledFunction)(q::AbstractVector{Int}) = f(indexof(q))
native_batch(::SampledFunction) = false
scalar_fastpath(::SampledFunction) = false
sampling_method(f::SampledFunction) = scalar_fastpath(f) ? "cached_scalar" :
    native_batch(f) ? "tt_slices" : "coordinate"

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
    map(f, slice_indices(f.R, left, right, middle))
end

struct TensorSamples <: SampledFunction
    evaluator::Evaluator
    frames::TCI.TTCache{ComplexF64}
    frame_capacity::Int
end
TensorSamples(e::Evaluator; frame_capacity = 2048) =
    TensorSamples(e, TCI.TTCache(e.x.c), frame_capacity)
sampled(x::QTT, rec) = TensorSamples(evaluate(x, rec);
    frame_capacity = get(rec.options, :slice_cache_capacity, 2048))
sample_length(f::TensorSamples) = length(f.evaluator.x)
native_batch(::TensorSamples) = true
scalar_fastpath(f::TensorSamples) =
    f.evaluator.prefix + f.evaluator.suffix == sample_length(f)
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
    scalar_fastpath(f) && return map(f.evaluator, slice_indices(sample_length(f), left, right, middle))
    result = TCI.batchevaluate(f.frames, left, right, middle)
    trim_frames!(f.frames, f.frame_capacity)
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
function (f::ShiftedSamples)(left, right, ::Val{M}) where M
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
        l = [vcat(bits(mod(indexof(left[i]) + f.offset, 1 << nl), nl), bits(center, M)) for i in rows]
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
    cache = SampleCache{ComplexF64}(R; capacity = options.scalar_cache_capacity,
        dense_limit = get(options, :dense_cache_max_points, 1 << 18))
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
function (f::RecordedSamples)(left, right, middle::Val{M}) where M
    f.batches += 1
    if scalar_fastpath(f)
        # Endpoint dot products are cheaper than grouping sparse misses and
        # building TTCache environments for every cross cut on small grids.
        before = f.evaluations
        result = map(f, slice_indices(f.R, left, right, middle))
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

mutable struct BlockOracle
    evals::Vector{Evaluator}
    samples::Vector{TensorSamples}
    cache::SampleCache{Vector{ComplexF64}}
    cap::Int
    requests::Int
    evaluations::Int
end
function BlockOracle(xs; cap = 8192, cachebits = 10, frame_capacity = 2048,
                     dense_limit = 0, max_bytes = 64 * 1024^2)
    evals = [Evaluator(x; cachebits) for x in xs]
    samples = [TensorSamples(e; frame_capacity) for e in evals]
    R = length(first(xs))
    # Include vector headers and references; values are populated only on demand.
    bytes_per_point = sizeof(ComplexF64) * length(evals) + 64
    limit = min(dense_limit, max_bytes ÷ bytes_per_point)
    cache = SampleCache{Vector{ComplexF64}}(R; capacity = cap, dense_limit = limit)
    BlockOracle(evals, samples, cache, isempty(cache.values) ? cap : length(cache.values), 0, 0)
end
scalar_fastpath(o::BlockOracle) = all(scalar_fastpath, o.samples)
function (o::BlockOracle)(j::Int)
    o.requests += 1
    value = cache_get(o.cache, j)
    if value === nothing
        value = cache_put!(o.cache, j, [e(j) for e in o.evals])
        o.evaluations += 1
    end
    value
end
function (o::BlockOracle)(left, right, middle::Val{M}) where M
    R = sample_length(first(o.samples))
    scalar_fastpath(o) && return map(o, slice_indices(R, left, right, middle))
    o.requests += length(left) * (1 << M) * length(right)
    result, evaluations, _ = cached_slice!(o.cache, R, left, right, middle) do l, r, m
        columns = [s(l, r, m) for s in o.samples]
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
    block_cache_capacity = o.cap, block_cache_evictions = o.cache.evictions,
    block_sample_requests = o.requests, block_sample_evaluations = o.evaluations)

struct LinearSamples <: SampledFunction
    oracle::BlockOracle
    weights::Vector{ComplexF64}
end
sample_length(f::LinearSamples) = sample_length(first(f.oracle.samples))
native_batch(::LinearSamples) = true
scalar_fastpath(f::LinearSamples) = scalar_fastpath(f.oracle)
function weighted_value(weights, values)
    z = zero(ComplexF64)
    @inbounds @simd for i in eachindex(weights, values)
        z += weights[i] * values[i]
    end
    z
end
(f::LinearSamples)(j::Int) = weighted_value(f.weights, f.oracle(j))
function (f::LinearSamples)(left, right, middle::Val)
    values = f.oracle(left, right, middle)
    isempty(values) && return zeros(ComplexF64, size(values))
    map(v -> weighted_value(f.weights, v), values)
end

struct DensitySamples <: SampledFunction
    oracle::BlockOracle
    weights::Vector{Float64}
end
sample_length(f::DensitySamples) = sample_length(first(f.oracle.samples))
native_batch(::DensitySamples) = true
scalar_fastpath(f::DensitySamples) = scalar_fastpath(f.oracle)
weighted_density(weights, values) = ComplexF64(sum(weights[i] * abs2(values[i]) for i in eachindex(weights, values)))
(f::DensitySamples)(j::Int) = weighted_density(f.weights, f.oracle(j))
function (f::DensitySamples)(left, right, middle::Val)
    map(v -> weighted_density(f.weights, v), f.oracle(left, right, middle))
end
