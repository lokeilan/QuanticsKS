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
