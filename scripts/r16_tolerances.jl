# One-dimensional tolerance experiment; every measured case starts from the same cores.
using QuanticsKS, LinearAlgebra, Random, JSON3, TOML, Dates

const K = QuanticsKS
const Q = K.Q
const ROOT = normpath(joinpath(@__DIR__, ".."))
length(ARGS) == 1 || error("Usage: r16_tolerances.jl NEW_OUTPUT_DIRECTORY")
const OUT = abspath(only(ARGS))
ispath(OUT) && error("Output already exists: $OUT")
mkpath(joinpath(OUT, "inputs"))
cp(@__FILE__, joinpath(OUT, basename(@__FILE__)))
cp(joinpath(ROOT, "inputs", "h100-r16.toml"), joinpath(OUT, "original-input.toml"))
BLAS.set_num_threads(1)

function json(path, value)
    open(io -> JSON3.pretty(io, value), path, "w")
end

base = TOML.parsefile(joinpath(OUT, "original-input.toml"))
base["run"]["blas_threads"] = 1
base["scf"]["parallel_tasks"] = 0
base["scf"]["max_iterations"] = 1
base["accuracy"]["density_tolerance"] = 1e-10
base["accuracy"]["xc_energy_tolerance"] = 1e-10
base["tci"]["post_rounding_factor"] = 0.1

function input(name, density_tol, post_factor; checkpoint = nothing)
    d = deepcopy(base)
    d["accuracy"]["density_tolerance"] = density_tol
    d["accuracy"]["xc_energy_tolerance"] = density_tol
    d["tci"]["post_rounding_factor"] = post_factor
    checkpoint === nothing || (d["run"]["checkpoint"] = checkpoint)
    path = joinpath(OUT, "inputs", name * ".toml")
    open(io -> TOML.print(io, d; sorted = true), path, "w")
    path
end

json(
    joinpath(OUT, "protocol.json"),
    (;
        system = "1D periodic 100-H chain",
        R = 16,
        occupied_orbitals = 50,
        original_input = "inputs/h100-r16.toml",
        seed = base["scf"]["seed"],
        julia_version = string(VERSION),
        julia_threads = Threads.nthreads(:default),
        interactive_threads = Threads.nthreads(:interactive),
        blas_threads = BLAS.get_num_threads(),
        initialization = "One cold baseline warm-up saves initial.jls; measured cases import a copy of those exact cores without pivot state. Warm-up is excluded from timing comparisons.",
        measured_scope = "SCF history row.seconds includes the first density/H/update/packing/output density/energy/residual; excludes setup, checkpoint writes and final audit. Whole configured runs are recorded separately.",
        audit_policy = "Retain each solver audit; additionally recompute every output with identical original accuracy, original TCI post-factor, default tighter AuditOptions and a reset seed, without imported pivots.",
        limitation = "One first SCF update cannot establish final convergence, physical acceptance or production speedup.",
        started_utc = string(now(UTC)),
    ),
)

println("Starting cold baseline warm-up at ", now())
flush(stdout)
warm_path = input("warmup", 1e-10, 0.1)
K.run_config(warm_path; output = joinpath(OUT, "warmup"))
mkpath(joinpath(OUT, "common-initial"))
common_path = joinpath(OUT, "common-initial", "initial.jls")
cp(joinpath(OUT, "warmup", "initial.jls"), common_path)

cases = [
    ("baseline", 1e-10, 0.1),
    ("density_xc", 1e-9, 0.1),
    ("post_svd", 1e-10, 0.3),
    ("combined", 1e-9, 0.3),
]
for (name, density_tol, post_factor) in cases
    GC.gc()
    println("Starting measured case ", name, " at ", now())
    flush(stdout)
    path = input(name, density_tol, post_factor; checkpoint = common_path)
    result = K.run_config(path; output = joinpath(OUT, name))
    println(
        "Completed ",
        name,
        ": first SCF seconds=",
        only(result.history).seconds,
        "; energy=",
        only(result.history).total,
        "; native audit=",
        result.audit.passed,
    )
    flush(stdout)
end

p = K.problem_from_config(base)
a = K.options_from_dict(Accuracy, base["accuracy"])
scf = K.options_from_dict(SCFOptions, base["scf"])
tci = K.options_from_dict(TCIOptions, base["tci"])
physics = PhysicsNumerics()
audit = AuditOptions()
initial = load_checkpoint(common_path)
initial_accuracy = Accuracy(; merge(K.option_tuple(a), (; density_tolerance = 1e-11))...)
Random.seed!(scf.seed)
initial_rec = K.recorder(nothing, tci; parallel_tasks = scf.parallel_tasks)
input_density = K.density(
    p,
    initial.orbitals,
    initial_accuracy,
    initial_rec;
    renormalize = true,
    label = "common_initial_density",
)
close(initial_rec)

for (name, _, _) in cases
    println("Starting common audit ", name, " at ", now())
    flush(stdout)
    GC.gc()
    Random.seed!(scf.seed)
    case_dir = joinpath(OUT, name)
    state = load_checkpoint(joinpath(case_dir, "checkpoint.jls"))
    status = JSON3.read(read(joinpath(case_dir, "status.json"), String))
    audit_dir = joinpath(case_dir, "common-audit")
    rec = K.recorder(audit_dir, tci; parallel_tasks = scf.parallel_tasks)
    rec.scf = 1
    rec.stage = "common_audit"
    started = time_ns()
    try
        check = K.audit_state(
            state,
            input_density,
            a,
            audit,
            physics,
            rec;
            reference_energy = status.final.total,
            scf,
        )
        json(
            joinpath(case_dir, "common-audit.json"),
            (;
                seconds = (time_ns() - started) / 1e9,
                audit = check,
                input_density = "Common initial cores fitted once at density tolerance 1e-11",
                tci = K.option_dict(tci),
                accuracy = K.option_dict(a),
                audit_options = K.option_dict(audit),
            ),
        )
    finally
        close(rec)
    end
end
json(
    joinpath(OUT, "experiment-status.json"),
    (; status = "finished", finished_utc = string(now(UTC))),
)
println("R16 first-SCF tolerance experiment finished at ", now())
