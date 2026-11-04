# Isolate single-inner threading from the existing parallel batch scheduling.
using QuanticsKS, LinearAlgebra, Random, JSON3, Statistics
const Q = QuanticsKS.Q
length(ARGS) == 2 || error("Usage: benchmark_inner.jl NEW_OUTPUT.json R16_CHECKPOINT")
const OUT = abspath(ARGS[1])
ispath(OUT) && error("Output already exists: $OUT")
BLAS.set_num_threads(1)
rng = MersenneTwister(20261104)
function train(cap)
    bonds = [min(2^min(k, 16 - k), cap) for k = 0:16]
    Q.normalize(
        Q.QTT([randn(rng, ComplexF64, bonds[k], 2, bonds[k + 1]) / sqrt(2bonds[k]) for k = 1:16]),
    )
end
function paired(f, g)
    f()
    g()
    old_bytes, new_bytes = @allocated(f()), @allocated(g())
    reps = min(200, max(4, fld(64 * 1024^2, max(1, old_bytes + new_bytes))))
    old_times, new_times = Float64[], Float64[]
    for trial = 1:8
        GC.gc()
        previous_gc = GC.enable(false)
        old_ns = new_ns = UInt64(0)
        try
            for repetition = 1:reps
                if isodd(trial + repetition)
                    start = time_ns()
                    f()
                    old_ns += time_ns() - start
                    start = time_ns()
                    g()
                    new_ns += time_ns() - start
                else
                    start = time_ns()
                    g()
                    new_ns += time_ns() - start
                    start = time_ns()
                    f()
                    old_ns += time_ns() - start
                end
            end
        finally
            GC.enable(previous_gc)
        end
        push!(old_times, old_ns / 1e9 / reps)
        push!(new_times, new_ns / 1e9 / reps)
    end
    (;
        serial_median_seconds = median(old_times),
        parallel_median_seconds = median(new_times),
        speedup = median(old_times) / median(new_times),
        serial_bytes = old_bytes,
        parallel_bytes = new_bytes,
        serial_seconds = old_times,
        parallel_seconds = new_times,
        repetitions = reps,
    )
end
state = load_checkpoint(ARGS[2])
cases = [("synthetic_cap_$cap", train(cap), train(cap)) for cap in (16, 32, 48)]
for i in (1, 25, 50)
    push!(cases, ("saved_R16_$i", state.orbitals[i], state.orbitals[mod1(i + 1, 50)]))
end
results = Any[]
for (name, x, y) in cases
    println("Benchmarking ", name)
    flush(stdout)
    serial() = Q.inner(x, y; cfg = (; parallel_tasks = 1))
    threaded() = Q.inner(x, y; cfg = (; parallel_tasks = 0))
    push!(
        results,
        merge(
            (;
                name,
                backend = Q.inner_backend(x, y),
                site_workers = Q.inner_direct_workers(x, y, (; parallel_tasks = 0)) > 1 ?
                               [
                    Q.inner_site_workers(gx, gy, (; parallel_tasks = 0)) for
                    (gx, gy) in zip(x.c, y.c)
                ] : ones(Int, length(x)),
                absolute_difference = abs(serial() - threaded()),
            ),
            paired(serial, threaded),
        ),
    )
end
report = (;
    scope = "Single direct-inner call with serial versus bounded internal Julia workers; excludes GC and process startup; outer orbital/Gram parallelism is checked separately.",
    julia_threads = Threads.nthreads(:default),
    blas_threads = BLAS.get_num_threads(),
    parallel_site_work_threshold = Q.DIRECT_INNER_PARALLEL_SITE_WORK,
    parallel_total_work_threshold = Q.DIRECT_INNER_PARALLEL_WORK,
    timing_policy = "Warm both paths; interleave calls and alternate order; GC before each batch, disabled inside; retain raw time series.",
    results,
)
mkpath(dirname(OUT))
open(io -> JSON3.pretty(io, report), OUT, "w")
println(JSON3.write(report))
