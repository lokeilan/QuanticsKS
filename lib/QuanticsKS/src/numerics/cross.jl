# Record TCI sampling, error estimates, and truncation; these estimates are not physical residuals.

function cross(
    r::Recorder,
    R,
    fn;
    tol = 1e-9,
    cap = 64,
    label = "",
    pivots = Int[],
    initial_zero_fallback = nothing,
)
    count = Ref(0)
    requests = Ref(0)
    cached = TCI.CachedFunction{ComplexF64}(fill(2, R)) do v
        count[] += 1
        ComplexF64(fn(indexof(v)))
    end
    function oracle(v)
        requests[] += 1
        if !(v in cached) && length(cached.cache) >= r.options.scalar_cache_capacity
            empty!(cached.cache)
        end
        cached(v)
    end
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
    r.calls += 1
    tic = time_ns()
    if all(j -> iszero(oracle(bits(j, R))), pts)
        if initial_zero_fallback !== nothing
            # Addition may cancel at every initial pivot. Use library algebra
            # rather than declaring the unsampled function globally zero.
            x = initial_zero_fallback()
            dt = (time_ns() - tic) / 1e9
            r.samples += count[]
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
                    oracle_evaluations = count[],
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
            r.samples += count[]
            r.tci_time += dt
            event!(
                r.tci_io,
                (
                    call = r.calls,
                    label = label,
                    seconds = dt,
                    oracle_evaluations = count[],
                    rank = 1,
                    reported_converged = true,
                    verified_zero = true,
                ),
            )
            return constant_tt(R, 0.0)
        end
        push!(pts, nonzero - 1)
    end
    tci, rs, err = TCI.crossinterpolate2(
        ComplexF64,
        oracle,
        fill(2, R),
        bits.(pts, R);
        tolerance = tol,
        maxbonddim = cap,
        maxiter = r.options.max_iterations,
        ncheckhistory = r.options.check_history,
        nsearchglobalpivot = r.options.global_pivot_search,
        maxnglobalpivot = r.options.max_global_pivots,
    )
    dt = (time_ns() - tic) / 1e9
    r.samples += count[]
    r.tci_time += dt
    rounding_start = time_ns()
    x = with_svd_context(;
        scf = r.scf,
        inner = r.inner,
        stage = r.stage,
        label,
        diagnostic_dir = r.dir === nothing ? nothing : joinpath(r.dir, "numerics-failures"),
    ) do
        rounded(QTT(TCI.TensorTrain(tci).sitetensors); tol = tol * r.options.post_rounding_factor, cap)
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
            oracle_evaluations = count[],
            requests = requests[],
            cache_peak_cap = r.options.scalar_cache_capacity,
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
    samples = oracle === nothing ? block_oracle(trains, rec) : j -> view(oracle(j), active)
    cross(
        rec,
        R,
        j -> sum(w * v for (w, v) in zip(weights, samples(j)));
        tol,
        cap,
        label,
        initial_zero_fallback = library_sum,
    )
end
