# Replay one fixed-H L1-L7 update; this is not a complete SCF performance study.
using QuanticsKS, LinearAlgebra, Random, JSON3, Statistics
const K = QuanticsKS
const Q = K.Q
length(ARGS) == 1 || error("Usage: benchmark_lrrap.jl NEW_OUTPUT.json")
output = abspath(only(ARGS))
ispath(output) && error("Output already exists: $output")
BLAS.set_num_threads(1)
Random.seed!(20261104)
p = hydrogen_chain(R = 8, natoms = 16)
a = Accuracy(orbital_cap = 24, field_cap = 32, work_cap = 48)
tci = TCIOptions(environment_cache_bits = 4, orbital_pivot_group_size = 2)
scf = SCFOptions(green_polish_every = 1)
cfg = K.solver_config(p, a, scf)
rec = K.recorder(nothing, tci)
ops = K.build_operators(p, a, rec, PhysicsNumerics(xc_table_points = 128))
state = K.initial_state(p, a, rec, InitialGuess())
H = K.effective_hamiltonian(p, K.density(p, state.orbitals, a, rec), ops, a, rec)
green = Q.green_mpo(p.grid.R, H.h, scf.green_shift)
close(rec)

function update(tasks)
    Random.seed!(20261104)
    localrec = K.recorder(nothing, tci; parallel_tasks = tasks)
    localrec.scf = localrec.inner = 1
    ys, ps, metrics = Q.lrrap_step(
        state.orbitals,
        nothing,
        H,
        green,
        merge(cfg, (; parallel_tasks = tasks)),
        localrec;
        work_tol = a.work_tolerance,
    )
    @assert metrics.green_polish && all(x -> Q.ttnorm(x) == 0, ps)
    calls, samples = localrec.calls, localrec.samples
    close(localrec)
    (; ys, metrics, calls, samples)
end

serial = update(1)
threaded = update(0)
function values(x)
    e = Q.Evaluator(x)
    [e(j) for j = 0:(2^length(x) - 1)]
end
errors = [norm(values(x) - values(y)) / norm(values(x)) for (x, y) in zip(serial.ys, threaded.ys)]
@assert maximum(errors) < 1e-8
@assert serial.calls == threaded.calls
serial_times, threaded_times = Float64[], Float64[]
for repetition = 1:3
    if isodd(repetition)
        push!(serial_times, @elapsed update(1))
        push!(threaded_times, @elapsed update(0))
    else
        push!(threaded_times, @elapsed update(0))
        push!(serial_times, @elapsed update(1))
    end
end
report = (;
    scope = "Warmed fixed-H L1-L7 replay including residual TCI, L2-L5, orbital reconstruction, Green polish and Ritz; excludes density feedback, storage packing, physical audit, run-log/checkpoint I/O and process startup",
    system = "1D periodic H16",
    R = p.grid.R,
    occupied_orbitals = length(p.occupations),
    julia_version = string(VERSION),
    julia_threads = Threads.nthreads(:default),
    blas_threads = BLAS.get_num_threads(),
    accuracy = K.option_dict(a),
    serial_seconds = serial_times,
    threaded_seconds = threaded_times,
    serial_median_seconds = median(serial_times),
    threaded_median_seconds = median(threaded_times),
    median_speedup = median(serial_times) / median(threaded_times),
    maximum_orbital_relative_l2_difference = maximum(errors),
    per_orbital_relative_l2_difference = errors,
    serial_tci_calls = serial.calls,
    threaded_tci_calls = threaded.calls,
    serial_tci_samples = serial.samples,
    threaded_tci_samples = threaded.samples,
    trace_after_difference_Ha = abs(serial.metrics.trace_after - threaded.metrics.trace_after),
    limitations = [
        "Not a complete SCF or production H100 benchmark.",
        "The comparison does not establish physical acceptance or peak memory.",
        "Both modes use the new deterministic batch RNG policy and skip discarded polish history compression.",
    ],
)
mkpath(dirname(output))
open(io -> JSON3.pretty(io, report), output, "w")
println(JSON3.write(report))
