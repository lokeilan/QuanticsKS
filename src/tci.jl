# Dense addressing on small grids; bounded CLOCK eviction on larger grids.
# A cache belongs to one fixed input function, never to a changing SCF label.
# TCI's cached functions retain all samples; this cache enforces the run's memory budget.
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
uses_batch_contraction(::SampledFunction) = false
uses_scalar_evaluation(::SampledFunction) = false
sampling_method(f::SampledFunction) =
    uses_scalar_evaluation(f) ? "cached_scalar" :
    uses_batch_contraction(f) ? "tt_slices" : "coordinate"

struct CoordinateSamples{F} <: SampledFunction
    f::F
    R::Int
end
(f::CoordinateSamples)(j::Int) = ComplexF64(f.f(j))
sampled(f, R::Int) = CoordinateSamples(f, R)
sampled(f::SampledFunction, R::Int) =
    sample_length(f) == R ? f : throw(DimensionMismatch("Sample grid"))
sample_length(f::CoordinateSamples) = f.R
uses_scalar_evaluation(::CoordinateSamples) = true
function (f::CoordinateSamples)(left, right, middle::Val)
    map_slice(f, ComplexF64, f.R, left, right, middle)
end

struct TensorSamples <: SampledFunction
    evaluator::Evaluator
    tt_cache::Union{Nothing,TCI.TTCache{ComplexF64}}
    tt_cache_capacity::Int
end
function TensorSamples(e::Evaluator; tt_cache_capacity = 2048)
    tt_cache = e.prefix + e.suffix == length(e.x) ? nothing : TCI.TTCache(e.x.c)
    TensorSamples(e, tt_cache, tt_cache_capacity)
end
sampled(x::QTT, rec) = TensorSamples(
    evaluate(x, rec);
    tt_cache_capacity = get(rec.options, :slice_cache_capacity, 2048),
)
sample_length(f::TensorSamples) = length(f.evaluator.x)
uses_batch_contraction(::TensorSamples) = true
uses_scalar_evaluation(f::TensorSamples) =
    f.evaluator.prefix + f.evaluator.suffix == sample_length(f)
(f::TensorSamples)(j::Int) = f.evaluator(j)

function trim_tt_cache!(tt_cache, capacity)
    caches = (tt_cache.cacheleft..., tt_cache.cacheright...)
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
    tt_cache = f.tt_cache
    tt_cache === nothing &&
        return map_slice(f.evaluator, ComplexF64, sample_length(f), left, right, middle)
    result = TCI.batchevaluate(tt_cache, left, right, middle)
    trim_tt_cache!(tt_cache, f.tt_cache_capacity)
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
uses_batch_contraction(f::PointwiseSamples) = any(uses_batch_contraction, f.arguments)
uses_scalar_evaluation(f::PointwiseSamples) = all(uses_scalar_evaluation, f.arguments)
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
uses_batch_contraction(f::ShiftedSamples) = uses_batch_contraction(f.input)
uses_scalar_evaluation(f::ShiftedSamples) = uses_scalar_evaluation(f.input)
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
uses_batch_contraction(f::RecordedSamples) = uses_batch_contraction(f.input)
uses_scalar_evaluation(f::RecordedSamples) = uses_scalar_evaluation(f.input)
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
    if uses_scalar_evaluation(f)
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

uses_tt_cache(::SampledFunction) = false
uses_tt_cache(f::TensorSamples) = f.tt_cache !== nothing
uses_tt_cache(f::PointwiseSamples) = any(uses_tt_cache, f.arguments)
uses_tt_cache(f::ShiftedSamples) = uses_tt_cache(f.input)
uses_tt_cache(f::RecordedSamples) = uses_tt_cache(f.input)

# The warm-start scale must see the current function on every old slice before
# optimize! starts. Stream endpoint values without allocating a result tensor.
function maximum_slice(f::RecordedSamples, left, right, middle::Val{M}) where {M}
    uses_scalar_evaluation(f) || return maximum(abs, f(left, right, middle); init = 0.0)
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

# Evaluate a block of tensor trains at shared grid indices.
# Scalar values and TCI batch contractions have separate bounded caches.
mutable struct BlockEvaluator
    evaluators::Vector{Evaluator}
    samples::Vector{TensorSamples}
    cache::SampleCache{Vector{ComplexF64}}
    cap::Int
    requests::Int
    evaluations::Int
    uses_endpoint_contraction::Bool
    parallel_tasks::Int
end
function BlockEvaluator(
    xs;
    cap = 8192,
    cachebits = 10,
    tt_cache_capacity = 2048,
    dense_limit = 0,
    max_bytes = 64 * 1024^2,
    parallel_tasks = 1,
)
    evaluators = [Evaluator(x; cachebits) for x in xs]
    samples = [TensorSamples(e; tt_cache_capacity) for e in evaluators]
    R = length(first(xs))
    # Include vector headers and references; values are populated only on demand.
    bytes_per_point = sizeof(ComplexF64) * length(evaluators) + 64
    limit = min(dense_limit, max_bytes ÷ bytes_per_point)
    cache = SampleCache{Vector{ComplexF64}}(R; capacity = cap, dense_limit = limit)
    BlockEvaluator(
        evaluators,
        samples,
        cache,
        isempty(cache.values) ? cap : length(cache.values),
        0,
        0,
        all(uses_scalar_evaluation, samples),
        parallel_tasks,
    )
end
uses_scalar_evaluation(o::BlockEvaluator) = o.uses_endpoint_contraction
function (o::BlockEvaluator)(j::Int)
    o.requests += 1
    value = cache_get(o.cache, j)
    if value === nothing
        value = cache_put!(o.cache, j, [e(j) for e in o.evaluators])
        o.evaluations += 1
    end
    value
end
function (o::BlockEvaluator)(left, right, middle::Val{M}) where {M}
    R = sample_length(first(o.samples))
    uses_scalar_evaluation(o) && return map_slice(o, Vector{ComplexF64}, R, left, right, middle)
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
block_cache_stats(o::BlockEvaluator) = (;
    block_cache_mode = isempty(o.cache.values) ? "bounded_clock" : "dense_addressed",
    block_cache_capacity = o.cap,
    block_cache_evictions = o.cache.evictions,
    block_sample_requests = o.requests,
    block_sample_evaluations = o.evaluations,
)

struct LinearSamples{O} <: SampledFunction
    evaluator::O
    weights::Vector{ComplexF64}
end
sample_length(f::LinearSamples) = sample_length(f.evaluator)
uses_batch_contraction(::LinearSamples) = true
uses_scalar_evaluation(f::LinearSamples) = uses_scalar_evaluation(f.evaluator)
# Linear-combination coefficients are not conjugated.
(f::LinearSamples)(j::Int) = transpose(f.weights) * f.evaluator(j)
function (f::LinearSamples)(left, right, middle::Val)
    values = f.evaluator(left, right, middle)
    isempty(values) && return zeros(ComplexF64, size(values))
    map(v -> transpose(f.weights) * v, values)
end

struct DensitySamples{O} <: SampledFunction
    evaluator::O
    weights::Vector{Float64}
end
sample_length(f::DensitySamples) = sample_length(f.evaluator)
uses_batch_contraction(::DensitySamples) = true
uses_scalar_evaluation(f::DensitySamples) = uses_scalar_evaluation(f.evaluator)
weighted_density(weights, values) =
    ComplexF64(sum(weights[i] * abs2(values[i]) for i in eachindex(weights, values)))
(f::DensitySamples)(j::Int) = weighted_density(f.weights, f.evaluator(j))
function (f::DensitySamples)(left, right, middle::Val)
    map(v -> weighted_density(f.weights, v), f.evaluator(left, right, middle))
end

sample_length(o::BlockEvaluator) = sample_length(first(o.samples))

struct SynchronizedSamples{F<:SampledFunction} <: SampledFunction
    input::F
    mutex::ReentrantLock
end
struct SynchronizedBlockEvaluator
    input::BlockEvaluator
    mutex::ReentrantLock
end
synchronized(f::SampledFunction) = SynchronizedSamples(f, ReentrantLock())
synchronized(f::BlockEvaluator) = SynchronizedBlockEvaluator(f, ReentrantLock())
synchronized(f::Union{SynchronizedSamples,SynchronizedBlockEvaluator}) = f
sample_length(f::Union{SynchronizedSamples,SynchronizedBlockEvaluator}) = sample_length(f.input)
uses_batch_contraction(f::SynchronizedSamples) = uses_batch_contraction(f.input)
uses_scalar_evaluation(f::Union{SynchronizedSamples,SynchronizedBlockEvaluator}) =
    uses_scalar_evaluation(f.input)
sampling_method(f::SynchronizedSamples) = sampling_method(f.input)
uses_tt_cache(f::SynchronizedSamples) = uses_tt_cache(f.input)
function (f::Union{SynchronizedSamples,SynchronizedBlockEvaluator})(j::Int)
    lock(f.mutex) do
        f.input(j)
    end
end
function (f::Union{SynchronizedSamples,SynchronizedBlockEvaluator})(left, right, middle::Val)
    lock(f.mutex) do
        f.input(left, right, middle)
    end
end
block_cache_stats(f::SynchronizedBlockEvaluator) = block_cache_stats(f.input)

# TCI fitting, pivot reuse, and fit records.

# Only pivot index sets survive a fit. Values, LU factors, and input environments are refreshed.
struct CrossState
    R::Int
    left::Vector{Vector{Vector{Int}}}
    right::Vector{Vector{Vector{Int}}}
    ranks::Vector{Int}
    errors::Vector{Float64}
    uses::Int
end
const CrossKey = Tuple{String,Int}

function remember_cross!(rec, keys::Tuple{Vararg{CrossKey}}, ci, ranks, errors)
    get(rec.options, :reuse_pivots, true) || return
    all(key -> isempty(first(key)), keys) && return
    # One read-only snapshot per fit. Each label retains its own usage count;
    # cross() copies the pivot index sets before TCI is allowed to modify it.
    left, right = deepcopy(ci.Iset), deepcopy(ci.Jset)
    saved_ranks, saved_errors = copy(ranks), copy(errors)
    for key in unique(keys)
        isempty(first(key)) && continue
        old = get(rec.cross_states, key, nothing)
        rec.cross_states[key] = CrossState(
            length(ci.localdims),
            left,
            right,
            saved_ranks,
            saved_errors,
            old === nothing ? 1 : old.uses + 1,
        )
        rec.cross_updates === nothing || push!(rec.cross_updates, key => rec.cross_states[key])
        filter!(!=(key), rec.cross_order)
        push!(rec.cross_order, key)
    end
    while length(rec.cross_order) > get(rec.options, :pivot_cache_capacity, 512)
        delete!(rec.cross_states, popfirst!(rec.cross_order))
    end
end
remember_cross!(rec, key::CrossKey, ci, ranks, errors) =
    remember_cross!(rec, (key,), ci, ranks, errors)

function cross_state_data(rec)
    rows = [
        Dict(
            "label" => key[1],
            "R" => s.R,
            "left" => s.left,
            "right" => s.right,
            "ranks" => s.ranks,
            "errors" => s.errors,
            "uses" => s.uses,
        ) for key in rec.cross_order for s in (rec.cross_states[key],)
    ]
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
            isempty(left[k]) ||
                length(unique(left[k])) == length(left[k]) ||
                throw(ArgumentError("Duplicate cross-state index"))
            isempty(right[k]) ||
                length(unique(right[k])) == length(right[k]) ||
                throw(ArgumentError("Duplicate cross-state index"))
            k < R &&
                length(right[k]) != length(left[k + 1]) &&
                throw(DimensionMismatch("Cross-state bond dimensions"))
        end
        first(left) == [Int[]] && last(right) == [Int[]] ||
            throw(ArgumentError("Cross-state boundaries"))
        state = CrossState(
            R,
            deepcopy(left),
            deepcopy(right),
            Int.(row["ranks"]),
            Float64.(row["errors"]),
            Int(row["uses"]),
        )
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

# Recorder tracks the stages, samples, and timings of TCI fits.

using JSON3, Dates, Printf

mutable struct Recorder
    options::NamedTuple
    dir::Union{Nothing,String}
    scf::Int
    inner::Int
    stage::String
    calls::Int
    samples::Int
    tci_time::Float64
    stage_io::IO
    tci_io::IO
    cross_states::Dict{CrossKey,CrossState}
    cross_order::Vector{CrossKey}
    cross_updates::Union{Nothing,Vector{Pair{CrossKey,CrossState}}}
end
function Recorder(dir; options)
    dir === nothing && return Recorder(
        options,
        nothing,
        0,
        0,
        "init",
        0,
        0,
        0.0,
        devnull,
        devnull,
        Dict{CrossKey,CrossState}(),
        CrossKey[],
        nothing,
    )
    mkpath(dir)
    Recorder(
        options,
        dir,
        0,
        0,
        "init",
        0,
        0,
        0.0,
        open(joinpath(dir, "stages.jsonl"), "w"),
        open(joinpath(dir, "tci_calls.jsonl"), "w"),
        Dict{CrossKey,CrossState}(),
        CrossKey[],
        nothing,
    )
end
function memory_info()
    if Sys.iswindows()
        v = zeros(UInt64, 10)
        v[1] = 80
        p = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
        ok = ccall(
            (:GetProcessMemoryInfo, "psapi"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt64}, UInt32),
            p,
            v,
            80,
        )
        ok != 0 && return (rss_bytes = v[3], peak_rss_bytes = v[2], private_bytes = v[10])
    end
    (rss_bytes = 0, peak_rss_bytes = Sys.maxrss(), private_bytes = 0)
end
function event!(io, row)
    io === devnull && return nothing
    JSON3.write(io, row)
    write(io, '\n')
    flush(io)
end
function timed(f, r::Recorder, id::String; item = "")
    prev = r.stage
    r.stage = id
    c = r.calls
    s = r.samples
    t = r.tci_time
    try
        z = @timed with_svd_context(
            f;
            scf = r.scf,
            inner = r.inner,
            stage = id,
            diagnostic_dir = r.dir === nothing ? nothing : joinpath(r.dir, "numerics-failures"),
        )
        event!(
            r.stage_io,
            merge(
                (
                    scf = r.scf,
                    inner = r.inner,
                    stage = id,
                    parent_stage = prev,
                    item = item,
                    seconds = z.time,
                    gc_seconds = z.gctime,
                    allocated_bytes = z.bytes,
                    tci_calls = r.calls - c,
                    tci_samples = r.samples - s,
                    tci_seconds = r.tci_time - t,
                    compile_seconds = hasproperty(z, :compile_time) ? z.compile_time : 0.0,
                ),
                memory_info(),
            ),
        )
        return z.value
    finally
        r.stage = prev
    end
end
function Base.close(r::Recorder)
    r.dir === nothing && return nothing
    try
        isempty(r.cross_states) || save_cross_state(r, joinpath(r.dir, "cross-state.jls"))
    finally
        close(r.stage_io)
        close(r.tci_io)
    end
end
debug_snapshot(r::Recorder, name, data) =
    r.dir === nothing ? nothing : serialize(joinpath(r.dir, name), data)

evaluate(x, r::Recorder) = Evaluator(x; cachebits = r.options.environment_cache_bits)
block_evaluator(xs, r::Recorder) = BlockEvaluator(
    xs;
    cap = r.options.orbital_cache_capacity,
    cachebits = r.options.environment_cache_bits,
    tt_cache_capacity = get(r.options, :slice_cache_capacity, 2048),
    dense_limit = get(r.options, :dense_cache_max_points, 262144),
    max_bytes = get(r.options, :orbital_cache_max_bytes, 64 * 1024^2),
    parallel_tasks = get(r.options, :parallel_tasks, 1),
)

# TCI construction and post-fit SVD rounding have separate tolerances and timings.

function cross(
    r::Recorder,
    R,
    fn;
    tol = 1e-9,
    cap = 64,
    label = "",
    pivots = Int[],
    shared_pivots = "",
    initial_zero_fallback = nothing,
    rounding_tolerance = iszero(r.options.post_rounding_factor) ? nothing :
                         tol * r.options.post_rounding_factor,
)
    validate_truncation(tol, cap)
    rounding_tolerance === nothing || validate_truncation(rounding_tolerance, cap)
    1 <= R <= 60 || throw(ArgumentError("Binary grid must have 1:60 sites"))
    evaluator = RecordedSamples(sampled(fn, R), r.options)
    key = (String(label), Int(R))
    shared_key = (String(shared_pivots), Int(R))
    saved =
        get(r.options, :reuse_pivots, true) && !isempty(label) ?
        get(r.cross_states, key, get(r.cross_states, shared_key, nothing)) : nothing
    saved !== nothing && maximum(length, saved.left) > cap && (saved = nothing)
    reuse_source =
        saved === nothing ? "none" : haskey(r.cross_states, key) ? "previous_fit" : "orbital_group"
    all(j -> 0 <= j < 2^R, r.options.initial_pivots) || throw(
        ArgumentError("TCI initial_pivots must be zero-based indices inside the current grid"),
    )
    pts = unique(
        vcat(
            pivots,
            r.options.initial_pivots,
            [0, 2^R - 1],
            rand(0:(2^R - 1), r.options.initial_random_pivots),
        ),
    )
    all(j -> 0 <= j < 2^R, pts) || throw(ArgumentError("Initial pivot outside the grid"))
    r.calls += 1
    call = r.calls
    tic = time_ns()
    initial_max = maximum(j -> abs(evaluator(j)), pts; init = 0.0)
    old_max =
        saved === nothing ? 0.0 :
        maximum(1:R) do k
            maximum_slice(evaluator, saved.left[k], saved.right[k], Val(1))
        end
    if iszero(max(initial_max, old_max))
        if initial_zero_fallback !== nothing
            # Zero initial samples do not establish global cancellation.
            # The caller supplies a reconstruction from the original cores.
            previous_tci_time = r.tci_time
            x = initial_zero_fallback()
            dt = (time_ns() - tic) / 1e9
            nested_tci_seconds = r.tci_time - previous_tci_time
            r.samples += evaluator.evaluations
            r.tci_time += max(0.0, dt - nested_tci_seconds)
            event!(
                r.tci_io,
                (;
                    call,
                    scf = r.scf,
                    inner = r.inner,
                    stage = r.stage,
                    label,
                    seconds = dt,
                    nested_tci_seconds,
                    function_evaluations = evaluator.evaluations,
                    rank = rank(x),
                    construction = "core_algebra_zero_initial_pivots",
                ),
            )
            return x
        end
        # TCI cannot initialize a zero pivot. Prove zero on bounded small grids;
        # never infer a globally zero function merely from a few zero samples.
        2^R <= r.options.zero_check_limit || error(
            "All initial TCI pivots are zero for $label. Supply informative pivots, an analytic constant potential, or increase zero_check_limit for a full scalar scan.",
        )
        nonzero = findfirst(j -> !iszero(evaluator(bits(j, R))), 0:(2^R - 1))
        if nonzero === nothing
            dt = (time_ns() - tic) / 1e9
            r.samples += evaluator.evaluations
            r.tci_time += dt
            event!(
                r.tci_io,
                (
                    call = call,
                    label = label,
                    seconds = dt,
                    function_evaluations = evaluator.evaluations,
                    rank = 1,
                    reported_converged = true,
                    verified_zero = true,
                ),
            )
            return constant_tt(R, 0.0)
        end
        push!(pts, nonzero - 1)
    end
    reused = saved !== nothing && old_max > 0
    tci = nothing
    if R == 1
        train = QTT([reshape(ComplexF64[evaluator(0), evaluator(1)], 1, 2, 1)])
        rs, err = [1], [0.0]
    else
        if reused
            tci = TCI.TensorCI2{ComplexF64}(fill(2, R))
            tci.Iset = deepcopy(saved.left)
            tci.Jset = deepcopy(saved.right)
            tci.maxsamplevalue = max(initial_max, old_max)
            TCI.addglobalpivots!(tci, bits.(pts, R))
            TCI.invalidatesitetensors!(tci)
        else
            tci = TCI.TensorCI2{ComplexF64}(evaluator, fill(2, R), bits.(pts, R))
        end
        rs, err = TCI.optimize!(
            tci,
            evaluator;
            tolerance = Float64(tol),
            maxbonddim = cap,
            maxiter = r.options.max_iterations,
            ncheckhistory = r.options.check_history,
            nsearchglobalpivot = r.options.global_pivot_search,
            maxnglobalpivot = r.options.max_global_pivots,
        )
        train = QTT(TCI.TensorTrain(tci).sitetensors)
    end
    dt = (time_ns() - tic) / 1e9
    r.samples += evaluator.evaluations
    r.tci_time += dt
    rounding_start = time_ns()
    x =
        rounding_tolerance === nothing ? train :
        with_svd_context(;
            scf = r.scf,
            inner = r.inner,
            stage = r.stage,
            label,
            diagnostic_dir = r.dir === nothing ? nothing : joinpath(r.dir, "numerics-failures"),
        ) do
            rounded(train; tol = rounding_tolerance, cap)
        end
    rounding_seconds = rounding_tolerance === nothing ? 0.0 : (time_ns() - rounding_start) / 1e9
    event!(
        r.tci_io,
        (
            call = call,
            scf = r.scf,
            inner = r.inner,
            stage = r.stage,
            label = label,
            construction = rounding_tolerance === nothing ? "tci_prrlu" :
                           "tci_prrlu_with_svd_rounding",
            seconds = dt,
            post_rounding_seconds = rounding_seconds,
            interpolation_and_rounding_seconds = dt + rounding_seconds,
            function_evaluations = evaluator.evaluations,
            requests = evaluator.requests,
            cache_mode = isempty(evaluator.cache.values) ? "bounded_clock" : "dense_addressed",
            cache_peak_cap = isempty(evaluator.cache.values) ? evaluator.cache.capacity :
                             length(evaluator.cache.values),
            cache_evictions = evaluator.cache.evictions,
            retained_sample_points = length(evaluator.cache),
            batch_calls = evaluator.batches,
            computed_batches = evaluator.computed_batches,
            batch_evaluations = evaluator.batch_evaluations,
            uses_batch_contraction = uses_batch_contraction(evaluator),
            sampling_method = sampling_method(evaluator),
            pivot_reused = reused,
            pivot_source = reused ? reuse_source : "cold",
            search_ranks = rs,
            search_errors = err,
            search_rank = rank(train),
            maxsamplevalue = tci === nothing ? max(initial_max, old_max) : tci.maxsamplevalue,
            tolerance = tol,
            rank_cap = cap,
            rank = rank(x),
            sweeps = length(rs),
            reported_error = isempty(err) ? 0.0 : last(err),
            reported_converged = !isempty(err) && last(err) <= tol,
            at_rank_cap = rank(x) >= cap,
            post_rounding_tolerance = rounding_tolerance,
        ),
    )
    r.options.require_convergence &&
        (isempty(err) || last(err) > tol) &&
        error("TCI did not meet tolerance: $label")
    if tci !== nothing
        remember_cross!(r, isempty(shared_pivots) ? (key,) : (key, shared_key), tci, rs, err)
    end
    x
end

"Evaluate a whole linear combination at each TCI query, with one reconstruction."
function linear_combination(
    rec::Recorder,
    xs,
    coefficients;
    tol = 1e-11,
    cap = 64,
    label = "linear_combination",
    evaluator = nothing,
    shared_pivots = "",
)
    validate_truncation(tol, cap)
    isempty(xs) && throw(ArgumentError("A QTT linear combination needs at least one term"))
    length(xs) == length(coefficients) || throw(DimensionMismatch("QTT coefficient count"))
    R = length(first(xs))
    all(x -> length(x) == R, xs) || throw(DimensionMismatch("QTT linear-combination lengths"))
    all(isfinite, coefficients) || throw(ArgumentError("Nonfinite QTT coefficients"))
    active = findall(!iszero, collect(coefficients))
    isempty(active) && return constant_tt(R, 0.0)
    trains = [xs[i] for i in active]
    weights = ComplexF64[coefficients[i] for i in active]
    if length(trains) == 1
        psi, coefficient = only(trains), only(weights)
        result = isone(coefficient) ? psi : scale(psi, coefficient)
        return rank(result) <= cap ? result : rounded(result; tol, cap)
    end
    # Combine all amplitudes before rounding so intermediate sums cannot lose cancellation.
    core_sum() = reconstruct_expression(linear_expression(trains, weights); tol, cap)
    R == 1 && return core_sum()
    samples = if evaluator === nothing
        LinearSamples(block_evaluator(trains, rec), weights)
    elseif evaluator isa Union{BlockEvaluator,SynchronizedBlockEvaluator}
        LinearSamples(evaluator, ComplexF64[coefficients...])
    else
        sampled(j -> sum(w * v for (w, v) in zip(weights, view(evaluator(j), active))), R)
    end
    cross(rec, R, samples; tol, cap, label, shared_pivots, initial_zero_fallback = core_sum)
end

# Fit batches own their recorder and RNG; fit records are merged in input order.

function fork_recorder(rec::Recorder)
    Recorder(
        rec.options,
        rec.dir,
        rec.scf,
        rec.inner,
        rec.stage,
        0,
        0,
        0.0,
        rec.stage_io === devnull ? devnull : IOBuffer(),
        rec.tci_io === devnull ? devnull : IOBuffer(),
        copy(rec.cross_states),
        copy(rec.cross_order),
        Pair{CrossKey,CrossState}[],
    )
end

function merge_recorder!(parent, child; sampling_seed = nothing, group = nothing)
    offset = parent.calls
    lines =
        child.tci_io === devnull ? String[] :
        split(String(take!(child.tci_io)), '\n'; keepempty = false)
    for line in lines
        row = Dict{String,Any}(String(k) => v for (k, v) in pairs(JSON3.read(line)))
        haskey(row, "call") && (row["call"] += offset)
        sampling_seed === nothing || get!(row, "sampling_seed", string(sampling_seed))
        group === nothing || get!(row, "parallel_group", group)
        event!(parent.tci_io, row)
    end
    if child.stage_io !== devnull
        write(parent.stage_io, take!(child.stage_io))
        flush(parent.stage_io)
    end
    parent.calls += child.calls
    parent.samples += child.samples
    parent.tci_time += child.tci_time
    for (key, state) in child.cross_updates
        old = get(parent.cross_states, key, nothing)
        updated = CrossState(
            state.R,
            state.left,
            state.right,
            state.ranks,
            state.errors,
            old === nothing ? 1 : old.uses + 1,
        )
        parent.cross_states[key] = updated
        parent.cross_updates === nothing || push!(parent.cross_updates, key => updated)
        filter!(!=(key), parent.cross_order)
        push!(parent.cross_order, key)
        while length(parent.cross_order) > get(parent.options, :pivot_cache_capacity, 512)
            delete!(parent.cross_states, popfirst!(parent.cross_order))
        end
    end
    nothing
end

"Fit independent groups from one pivot snapshot and merge logs in input order."
function recorded_map(
    f,
    ::Type{T},
    n::Integer,
    cfg,
    rec::Recorder;
    group_size = 1,
    random = true,
) where {T}
    n >= 0 && group_size > 0 || throw(ArgumentError("Invalid recorded batch dimensions"))
    n == 0 && return T[]
    groups = cld(n, group_size)
    seeds = random ? rand(UInt64, groups) : fill(UInt64(0), groups)
    children = [fork_recorder(rec) for _ = 1:groups]
    result = Vector{T}(undef, n)
    failures = fill!(Vector{Any}(undef, groups), nothing)
    cancelled = Threads.Atomic{Bool}(false)
    foreach_independent(groups, cfg) do group
        cancelled[] && return
        child = children[group]
        first = (group - 1) * group_size + 1
        last = min(group * group_size, n)
        try
            run = () -> begin
                for i = first:last
                    result[i] = f(i, child)
                end
            end
            with_svd_context(;
                parallel_group = group,
                sampling_seed = random ? string(seeds[group]) : nothing,
            ) do
                random ? with_sampling_seed(run, seeds[group]) : run()
            end
        catch exception
            cancelled[] = true
            failures[group] = exception
            event!(
                child.stage_io,
                (;
                    scf = child.scf,
                    inner = child.inner,
                    stage = child.stage,
                    event = "parallel_group_failure",
                    first_item = first,
                    last_item = last,
                    error = sprint(showerror, exception),
                    seconds = 0.0,
                    parent_stage = rec.stage,
                    tci_calls = 0,
                    tci_samples = 0,
                    tci_seconds = 0.0,
                ),
            )
        end
    end
    # Even a failed group retains its completed fits and diagnostic events.
    for (group, child) in enumerate(children)
        merge_recorder!(rec, child; sampling_seed = random ? seeds[group] : nothing, group)
    end
    failed = findfirst(!isnothing, failures)
    failed === nothing || throw(failures[failed])
    event!(
        rec.stage_io,
        (;
            scf = rec.scf,
            inner = rec.inner,
            stage = rec.stage,
            event = "parallel_batch",
            seconds = 0.0,
            parent_stage = rec.stage,
            items = n,
            groups,
            group_size,
            workers = parallel_workers(cfg, groups),
            tci_calls = 0,
            tci_samples = 0,
            tci_seconds = 0.0,
            sampling_rng = random ? "ordered_group_seeds" : "unused",
        ),
    )
    result
end

# Task scheduling and sampling RNG scope.

# Bound independent work; nested contractions reuse the active worker.

const PARALLEL_WORK_KEY = :quanticsks_parallel_work
const PARALLEL_BUDGET_KEY = :quanticsks_parallel_budget

function independent_worker(f, worker, workers, n, context)
    with_svd_context(; context...) do
        tls = task_local_storage()
        previous = get(tls, PARALLEL_WORK_KEY, false)
        tls[PARALLEL_WORK_KEY] = true
        try
            for i = worker:workers:n
                f(i)
            end
        finally
            previous ? (tls[PARALLEL_WORK_KEY] = true) : delete!(tls, PARALLEL_WORK_KEY)
        end
    end
end

function parallel_workers(cfg, n::Integer)
    requested = hasproperty(cfg, :parallel_tasks) ? cfg.parallel_tasks : 1
    requested isa Integer && requested >= 0 ||
        throw(ArgumentError("parallel_tasks must be a nonnegative integer"))
    available = min(
        Threads.nthreads(:default),
        get(task_local_storage(), PARALLEL_BUDGET_KEY, Threads.nthreads(:default)),
    )
    limit = get(task_local_storage(), PARALLEL_WORK_KEY, false) ? 1 : available
    min(n, limit, requested == 0 ? available : requested)
end

"Run independent indexed work with bounded concurrency and inherited SVD diagnostics."
function foreach_independent(f, n::Integer, cfg)
    n >= 0 || throw(ArgumentError("Negative independent work count"))
    workers = parallel_workers(cfg, n)
    if workers <= 1
        tls = task_local_storage()
        previous = get(tls, PARALLEL_BUDGET_KEY, nothing)
        requested = hasproperty(cfg, :parallel_tasks) ? cfg.parallel_tasks : 1
        tls[PARALLEL_BUDGET_KEY] = min(
            something(previous, Threads.nthreads(:default)),
            requested == 0 ? Threads.nthreads(:default) : requested,
        )
        try
            for i = 1:n
                f(i)
            end
        finally
            previous === nothing ? delete!(tls, PARALLEL_BUDGET_KEY) :
            (tls[PARALLEL_BUDGET_KEY] = previous)
        end
    else
        context = get(task_local_storage(), SVD_CONTEXT_KEY, (;))
        @sync for worker = 1:workers
            Threads.@spawn :default independent_worker(f, worker, workers, n, context)
        end
    end
    nothing
end

function independent_map(f, ::Type{T}, n::Integer, cfg) where {T}
    result = Vector{T}(undef, n)
    foreach_independent(n, cfg) do i
        result[i] = f(i)
    end
    result
end

function with_sampling_seed(f, seed)
    saved = copy(Random.default_rng())
    try
        Random.seed!(seed)
        f()
    finally
        copy!(Random.default_rng(), saved)
    end
end

"Write disjoint columns directly, preserving their order and avoiding a second block copy."
function independent_columns(f, rows::Integer, columns::Integer, cfg)
    result = Matrix{ComplexF64}(undef, rows, columns)
    foreach_independent(columns, cfg) do j
        result[:, j] = f(j)
    end
    result
end
