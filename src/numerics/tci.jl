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

function remember_cross!(rec, keys::Tuple{Vararg{CrossKey}}, ci, ranks, errors)
    get(rec.options, :reuse_pivots, true) || return
    all(key -> isempty(first(key)), keys) && return
    # One read-only snapshot per fit. Each label retains its own usage count;
    # cross() copies the geometry before TCI is allowed to modify it.
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

# Recorder tracks stages, samples, and timings; cross.jl performs TCI fits.

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
block_oracle(xs, r::Recorder) = BlockOracle(
    xs;
    cap = r.options.orbital_cache_capacity,
    cachebits = r.options.environment_cache_bits,
    frame_capacity = get(r.options, :slice_cache_capacity, 2048),
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
    algebra_options(tol, cap)
    rounding_tolerance === nothing || algebra_options(rounding_tolerance, cap)
    1 <= R <= 60 || throw(ArgumentError("Binary grid must have 1:60 sites"))
    oracle = RecordedSamples(sampled(fn, R), r.options)
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
    tic = time_ns()
    initial_max = maximum(j -> abs(oracle(j)), pts; init = 0.0)
    old_max =
        saved === nothing ? 0.0 : maximum(1:R) do k
            maximum_slice(oracle, saved.left[k], saved.right[k], Val(1))
        end
    if iszero(max(initial_max, old_max))
        if initial_zero_fallback !== nothing
            # Addition may cancel at every initial pivot. Use library algebra
            # rather than declaring the unsampled function globally zero.
            x = initial_zero_fallback()
            dt = (time_ns() - tic) / 1e9
            r.samples += oracle.evaluations
            r.tci_time += dt
            event!(
                r.tci_io,
                (;
                    call = r.calls,
                    scf = r.scf,
                    inner = r.inner,
                    stage = r.stage,
                    label,
                    seconds = dt,
                    oracle_evaluations = oracle.evaluations,
                    rank = rank(x),
                    construction = "library_tt_add_zero_initial_pivots",
                ),
            )
            return x
        end
        # TCI cannot initialize a zero pivot. Prove zero on bounded small grids;
        # never infer a globally zero function merely from a few zero samples.
        2^R <= r.options.zero_check_limit || error(
            "All initial TCI pivots are zero for $label. Supply informative pivots, an analytic constant potential, or increase zero_check_limit for a full scalar scan.",
        )
        nonzero = findfirst(j -> !iszero(oracle(bits(j, R))), 0:(2^R - 1))
        if nonzero === nothing
            dt = (time_ns() - tic) / 1e9
            r.samples += oracle.evaluations
            r.tci_time += dt
            event!(
                r.tci_io,
                (
                    call = r.calls,
                    label = label,
                    seconds = dt,
                    oracle_evaluations = oracle.evaluations,
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
        train = QTT([reshape(ComplexF64[oracle(0), oracle(1)], 1, 2, 1)])
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
            tci = TCI.TensorCI2{ComplexF64}(oracle, fill(2, R), bits.(pts, R))
        end
        rs, err = TCI.optimize!(
            tci,
            oracle;
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
    r.samples += oracle.evaluations
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
            call = r.calls,
            scf = r.scf,
            inner = r.inner,
            stage = r.stage,
            label = label,
            construction = rounding_tolerance === nothing ? "tci_prrlu" :
                           "tci_prrlu_with_svd_rounding",
            seconds = dt,
            post_rounding_seconds = rounding_seconds,
            interpolation_and_rounding_seconds = dt + rounding_seconds,
            oracle_evaluations = oracle.evaluations,
            requests = oracle.requests,
            cache_mode = isempty(oracle.cache.values) ? "bounded_clock" : "dense_addressed",
            cache_peak_cap = isempty(oracle.cache.values) ? oracle.cache.capacity :
                             length(oracle.cache.values),
            cache_evictions = oracle.cache.evictions,
            retained_sample_points = length(oracle.cache),
            batch_calls = oracle.batches,
            computed_batches = oracle.computed_batches,
            batch_evaluations = oracle.batch_evaluations,
            native_batch = native_batch(oracle),
            sampling_method = sampling_method(oracle),
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
    oracle = nothing,
    shared_pivots = "",
)
    algebra_options(tol, cap)
    isempty(xs) && throw(ArgumentError("A QTT linear combination needs at least one term"))
    length(xs) == length(coefficients) || throw(DimensionMismatch("QTT coefficient count"))
    R = length(first(xs))
    all(x -> length(x) == R, xs) || throw(DimensionMismatch("QTT linear-combination lengths"))
    all(isfinite, coefficients) || throw(ArgumentError("Nonfinite QTT coefficients"))
    active = findall(!iszero, collect(coefficients))
    isempty(active) && return constant_tt(R, 0.0)
    trains = [xs[i] for i in active]
    weights = ComplexF64[coefficients[i] for i in active]
    length(trains) == 1 && return rounded(scale(only(trains), only(weights)); tol, cap)
    function library_sum()
        result = scale(first(trains), first(weights))
        for i = 2:length(trains)
            result = add(result, trains[i]; b = weights[i], tol, cap)
        end
        result
    end
    R == 1 && return library_sum()
    samples = if oracle === nothing
        LinearSamples(block_oracle(trains, rec), weights)
    elseif oracle isa Union{BlockOracle,SynchronizedBlockOracle}
        LinearSamples(oracle, ComplexF64[coefficients...])
    else
        sampled(j -> sum(w * v for (w, v) in zip(weights, view(oracle(j), active))), R)
    end
    cross(rec, R, samples; tol, cap, label, shared_pivots, initial_zero_fallback = library_sum)
end

# Fit batches own their recorder and RNG; only ordered receipts reach the parent.

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
