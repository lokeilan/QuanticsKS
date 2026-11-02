# Keep the endpoint scalar fast path; use the library's matrix contractions for slices.
abstract type SampledFunction <: TCI.BatchEvaluator{ComplexF64} end
(f::SampledFunction)(q::AbstractVector{Int}) = f(indexof(q))
native_batch(::SampledFunction) = false

struct CoordinateSamples{F} <: SampledFunction
    f::F
    R::Int
end
(f::CoordinateSamples)(j::Int) = ComplexF64(f.f(j))
sampled(f, R::Int) = CoordinateSamples(f, R)
sampled(f::SampledFunction, R::Int) =
    sample_length(f) == R ? f : throw(DimensionMismatch("Sample grid"))
sample_length(f::CoordinateSamples) = f.R
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

struct BlockOracle
    evals::Vector{Evaluator}
    samples::Vector{TensorSamples}
    cache::SampleCache{Vector{ComplexF64}}
    cap::Int
end
function BlockOracle(xs; cap = 8192, cachebits = 10, frame_capacity = 2048)
    evals = [Evaluator(x; cachebits) for x in xs]
    samples = [TensorSamples(e; frame_capacity) for e in evals]
    BlockOracle(evals, samples, SampleCache{Vector{ComplexF64}}(length(first(xs)); capacity = cap), cap)
end
function (o::BlockOracle)(j::Int)
    value = cache_get(o.cache, j)
    value === nothing ? cache_put!(o.cache, j, [e(j) for e in o.evals]) : value
end
function (o::BlockOracle)(left, right, middle::Val)
    result, _, _ = cached_slice!(o.cache, sample_length(first(o.samples)), left, right, middle) do l, r, m
        columns = [s(l, r, m) for s in o.samples]
        out = Array{Vector{ComplexF64}}(undef, size(first(columns)))
        for i in eachindex(out)
            out[i] = ComplexF64[c[i] for c in columns]
        end
        out
    end
    result
end

struct LinearSamples <: SampledFunction
    oracle::BlockOracle
    weights::Vector{ComplexF64}
end
sample_length(f::LinearSamples) = sample_length(first(f.oracle.samples))
native_batch(::LinearSamples) = true
(f::LinearSamples)(j::Int) = sum(f.weights .* f.oracle(j))
function (f::LinearSamples)(left, right, middle::Val)
    values = f.oracle(left, right, middle)
    isempty(values) && return zeros(ComplexF64, size(values))
    # Columns of A are the input orbitals; ordinary transpose preserves complex amplitudes.
    A = permutedims(reduce(hcat, vec(values)))
    reshape(A * f.weights, size(values))
end

struct DensitySamples <: SampledFunction
    oracle::BlockOracle
    weights::Vector{Float64}
end
sample_length(f::DensitySamples) = sample_length(first(f.oracle.samples))
native_batch(::DensitySamples) = true
(f::DensitySamples)(j::Int) = ComplexF64(sum(f.weights .* abs2.(f.oracle(j))))
function (f::DensitySamples)(left, right, middle::Val)
    map(v -> ComplexF64(sum(f.weights .* abs2.(v))), f.oracle(left, right, middle))
end
