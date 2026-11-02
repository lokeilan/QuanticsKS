# Record TCI sampling, error estimates, and truncation; these estimates are not physical residuals.

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
)
    algebra_options(tol, cap)
    1 <= R <= 60 || throw(ArgumentError("Binary grid must have 1:60 sites"))
    oracle = RecordedSamples(sampled(fn, R), r.options)
    key = (String(label), Int(R))
    shared_key = (String(shared_pivots), Int(R))
    saved = get(r.options, :reuse_pivots, true) && !isempty(label) ?
        get(r.cross_states, key, get(r.cross_states, shared_key, nothing)) : nothing
    saved !== nothing && maximum(length, saved.left) > cap && (saved = nothing)
    reuse_source = saved === nothing ? "none" : haskey(r.cross_states, key) ? "previous_fit" : "orbital_group"
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
    old_max = saved === nothing ? 0.0 : maximum(1:R) do k
        maximum(abs, oracle(saved.left[k], saved.right[k], Val(1)); init = 0.0)
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
        rs, err = TCI.optimize!(tci, oracle;
            tolerance = Float64(tol), maxbonddim = cap,
            maxiter = r.options.max_iterations, ncheckhistory = r.options.check_history,
            nsearchglobalpivot = r.options.global_pivot_search,
            maxnglobalpivot = r.options.max_global_pivots)
        train = QTT(TCI.TensorTrain(tci).sitetensors)
    end
    dt = (time_ns() - tic) / 1e9
    r.samples += oracle.evaluations
    r.tci_time += dt
    rounding_start = time_ns()
    x = with_svd_context(;
        scf = r.scf,
        inner = r.inner,
        stage = r.stage,
        label,
        diagnostic_dir = r.dir === nothing ? nothing : joinpath(r.dir, "numerics-failures"),
    ) do
        rounded(train; tol = tol * r.options.post_rounding_factor, cap)
    end
    rounding_seconds = (time_ns() - rounding_start) / 1e9
    event!(
        r.tci_io,
        (
            call = r.calls,
            scf = r.scf,
            inner = r.inner,
            stage = r.stage,
            label = label,
            seconds = dt,
            post_rounding_seconds = rounding_seconds,
            interpolation_and_rounding_seconds = dt + rounding_seconds,
            oracle_evaluations = oracle.evaluations,
            requests = oracle.requests,
            cache_mode = isempty(oracle.cache.values) ? "bounded_clock" : "dense_addressed",
            cache_peak_cap = isempty(oracle.cache.values) ? oracle.cache.capacity : length(oracle.cache.values),
            cache_evictions = oracle.cache.evictions,
            retained_sample_points = length(oracle.cache),
            batch_calls = oracle.batches,
            computed_batches = oracle.computed_batches,
            batch_evaluations = oracle.batch_evaluations,
            native_batch = native_batch(oracle),
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
            post_rounding_tolerance = tol * r.options.post_rounding_factor,
        ),
    )
    r.options.require_convergence &&
        (isempty(err) || last(err) > tol) &&
        error("TCI did not meet tolerance: $label")
    if tci !== nothing
        remember_cross!(r, key, tci, rs, err)
        isempty(shared_pivots) || remember_cross!(r, shared_key, tci, rs, err)
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
    elseif oracle isa BlockOracle
        LinearSamples(oracle, ComplexF64[coefficients...])
    else
        sampled(j -> sum(w * v for (w, v) in zip(weights, view(oracle(j), active))), R)
    end
    cross(
        rec,
        R,
        samples;
        tol,
        cap,
        label,
        shared_pivots,
        initial_zero_fallback = library_sum,
    )
end
