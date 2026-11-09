# Validate options and manage run logs; collect source snapshots from this src directory.

option_dict(x) = Dict(string(k) => getfield(x, k) for k in fieldnames(typeof(x)))
option_tuple(x) = (; (k => getfield(x, k) for k in fieldnames(typeof(x)))...)
function validate(t::TCIOptions)
    min(
        t.max_iterations,
        t.check_history,
        t.scalar_cache_capacity,
        t.orbital_cache_capacity,
        t.slice_cache_capacity,
        t.pivot_cache_capacity,
        t.orbital_pivot_group_size,
        t.zero_check_limit,
    ) > 0 || throw(ArgumentError("Invalid TCI counts"))
    t.dense_cache_max_points >= 0 || throw(ArgumentError("Negative dense cache limit"))
    t.orbital_cache_max_bytes >= 0 || throw(ArgumentError("Negative orbital cache budget"))
    min(t.global_pivot_search, t.max_global_pivots, t.initial_random_pivots) >= 0 ||
        throw(ArgumentError("Negative pivot count"))
    0 <= t.environment_cache_bits <= 16 ||
        throw(ArgumentError("Environment cache bits must be 0:16"))
    isfinite(t.post_rounding_factor) && 0 <= t.post_rounding_factor <= 1 ||
        throw(ArgumentError("Invalid post-rounding factor"))
    t
end
function recorder(dir, t::TCIOptions; parallel_tasks = 1)
    parallel_tasks isa Integer && parallel_tasks >= 0 ||
        throw(ArgumentError("parallel_tasks must be a nonnegative integer"))
    Q.Recorder(
        dir;
        options = merge(option_tuple(validate(t)), (; parallel_tasks = Int(parallel_tasks))),
    )
end
function validate(r::RefinementOptions)
    r.degree in (1, 3, 5, 7, 9) &&
        1 <= r.max_added_bits <= 12 &&
        isfinite(r.tolerance) &&
        r.tolerance > 0 || throw(ArgumentError("Invalid refinement options"))
    r
end
function validate(p::PhysicsNumerics)
    0 < p.green_relative_tolerance < 1 && p.green_max_iterations > 0 ||
        throw(ArgumentError("Invalid 3D Green solve controls"))
    p.xc_table_points >= 4 &&
        0 < p.xc_density_min < p.xc_density_max &&
        isfinite(p.xc_density_max) &&
        0 < p.exchange_quadrature_rtol < 1 || throw(ArgumentError("Invalid XC numerics"))
    p
end
function validate(g::InitialGuess)
    all(isfinite, (g.envelope_amplitude, g.slow_cosine_amplitude, g.slow_sine_amplitude)) ||
        throw(ArgumentError("Initial amplitudes must be finite"))
    g.envelope_period === nothing ||
        (isfinite(g.envelope_period) && g.envelope_period > 0) ||
        throw(ArgumentError("Invalid envelope period"))
    g
end
function validate(a::AuditOptions)
    a.rank_multiplier >= 1 && all(
        x -> isfinite(x) && x > 0,
        (
            a.field_tolerance,
            a.kernel_tolerance,
            a.algebra_tolerance,
            a.gram_tolerance,
            a.charge_tolerance,
        ),
    ) || throw(ArgumentError("Invalid audit options"))
    a
end
function validate(s::SCFOptions)
    s.solver in (:lrrap, SUBSPACE_SOLVERS...) || throw(
        ArgumentError(
            "Unknown SCF solver $(s.solver); choose :lrrap, :lanczos, :davidson or :block_davidson",
        ),
    )
    s.parallel_tasks >= 0 || throw(ArgumentError("parallel_tasks must be nonnegative"))
    s.criterion in (:energy, :strict) ||
        throw(ArgumentError("SCF criterion must be :energy or :strict"))
    min(s.max_iterations, s.inner_steps, s.coefficient_sweeps, s.anderson_history) > 0 &&
        0 < s.mixing <= 1 || throw(ArgumentError("Invalid SCF iteration/mixing settings"))
    s.green_update_interval >= 0 || throw(ArgumentError("Negative Green update interval"))
    for key in (
        :energy_tolerance,
        :eigen_tolerance,
        :density_tolerance,
        :green_shift,
        :anderson_regularization,
        :anderson_coefficient_limit,
        :tangent_relative_cutoff,
        :tangent_absolute_cutoff,
        :metric_relative_cutoff,
        :metric_absolute_cutoff,
        :constraint_relative_cutoff,
        :constraint_absolute_cutoff,
        :orthogonalize_threshold,
        :independence_threshold,
        :gram_tolerance,
    )
        value = getfield(s, key)
        isfinite(value) && value > 0 || throw(ArgumentError("Invalid SCF $key"))
    end
    isfinite(s.green_update_residual_threshold) && s.green_update_residual_threshold >= 0 ||
        throw(ArgumentError("Invalid Green update residual threshold"))
    s
end
function prepare_run(output, kind, p; options...)
    output === nothing && return nothing
    dir = abspath(output)
    ispath(dir) && throw(ArgumentError("Run output already exists: $dir; choose a fresh directory"))
    mkpath(joinpath(dir, "snapshots"))
    files = sort([
        joinpath(root, f) for (root, _, fs) in walkdir(@__DIR__) for f in fs if endswith(f, ".jl")
    ])
    for f in files
        dest = joinpath(dir, "source", "src", relpath(f, @__DIR__))
        mkpath(dirname(dest))
        cp(f, dest)
    end
    for name in ("Project.toml", "Manifest.toml", "run.jl")
        src = joinpath(@__DIR__, "..", name)
        isfile(src) && cp(src, joinpath(dir, "source", name))
    end
    config = Dict(
        "kind" => kind,
        "problem" => problem_descriptor(p),
        "options" => Dict(string(k) => option_dict(v) for (k, v) in options),
        "started_utc" => string(now(UTC)),
        "julia_version" => string(VERSION),
        "cpu" => Sys.CPU_NAME,
        "blas_threads" => BLAS.get_num_threads(),
        "julia_threads" => Threads.nthreads(:default),
        "julia_interactive_threads" => Threads.nthreads(:interactive),
        "sampling_rng_policy" => "ordered group seeds for recorded batches; serial and threaded modes share the policy",
        "parallel_timing_policy" => "stage seconds are wall time; summed fit/substage seconds may overlap across workers and must not be added to parent wall time",
        "timing_scope" => "in-process end-to-end wall time; includes initialization, diagnostics and checkpoint I/O; excludes Julia process startup and later postprocessing",
    )
    haskey(options, :accuracy) && (config["resolved_field_caps"] = field_caps(options[:accuracy]))
    write_json(joinpath(dir, "config.json"), config)
    dir
end
write_json(path, data) = open(io -> JSON3.write(io, data), path, "w")
run_io(dir, name) = dir === nothing ? devnull : open(joinpath(dir, name), "w")
close_run_io(io) = io === devnull ? nothing : close(io)
function fail_run(dir, e)
    dir === nothing ||
        write_json(joinpath(dir, "status.json"), (status = "failed", error = sprint(showerror, e)))
end

# Convert TOML into models and options, reporting input errors before solving.

"Construct typed options, rejecting unknown keys rather than silently ignoring typos."
function options_from_dict(::Type{T}, d) where {T}
    fields = fieldnames(T)
    pairs = Pair{Symbol,Any}[]
    for (k, v) in d
        key = Symbol(k)
        key in fields || throw(ArgumentError("Unknown $T option: $k"))
        fieldtype(T, key) == Symbol && (v = Symbol(v))
        push!(pairs, key => v)
    end
    T(; pairs...)
end
function require_keys(d, allowed, section)
    bad = setdiff(collect(keys(d)), allowed)
    isempty(bad) || throw(ArgumentError("Unknown $section keys: $(join(bad,", "))"))
end
function problem_from_config(d)
    gd = get(d, "grid", Dict())
    require_keys(gd, ["R", "cell", "dimensions"], "grid")
    sd = get(d, "system", Dict())
    kind = get(sd, "kind", "hydrogen_chain")
    dimensions = get(gd, "dimensions", 1)
    dimensions == 3 && return problem_from_config3d(gd, sd)
    dimensions == 1 || throw(ArgumentError("Grid dimensions must be 1 or 3"))
    common = ["kind", "mass", "occupations"]
    if kind == "hydrogen_chain"
        require_keys(
            sd,
            vcat(common, ["natoms", "distance", "offset", "softening", "coupling", "xc"]),
            "system",
        )
        n = get(sd, "natoms", 100)
        distance = get(sd, "distance", 4.0)
        haskey(sd, "occupations") && throw(
            ArgumentError(
                "hydrogen_chain uses double occupations; use soft_coulomb for explicit occupations",
            ),
        )
        return hydrogen_chain(
            R = get(gd, "R", 16),
            natoms = n,
            distance = distance,
            cell = get(gd, "cell", n * distance),
            offset = get(sd, "offset", 0.0),
            softening = get(sd, "softening", 1.0),
            coupling = get(sd, "coupling", 1.0),
            xc = Symbol(get(sd, "xc", "lsda")),
            mass = get(sd, "mass", 1.0),
        )
    elseif kind == "soft_coulomb"
        require_keys(
            sd,
            vcat(common, ["positions", "charges", "softening", "coupling", "xc"]),
            "system",
        )
        m = SoftCoulombLSDA1D(
            sd["positions"],
            sd["charges"];
            softening = get(sd, "softening", 1.0),
            coupling = get(sd, "coupling", 1.0),
            xc = Symbol(get(sd, "xc", "lsda")),
        )
        return Problem(
            PeriodicGrid1D(gd["R"], gd["cell"]),
            m,
            sd["occupations"];
            mass = get(sd, "mass", 1.0),
        )
    elseif kind == "cosine_potential"
        require_keys(sd, vcat(common, ["amplitude", "mode", "phase", "offset"]), "system")
        L = gd["cell"]
        amplitude = get(sd, "amplitude", 0.0)
        mode = get(sd, "mode", 1)
        phase = get(sd, "phase", 0.0)
        offset = get(sd, "offset", 0.0)
        all(isfinite, (amplitude, phase, offset)) && mode isa Integer ||
            throw(ArgumentError("Invalid cosine potential"))
        id = "cosine:L=$L:a=$amplitude:m=$mode:p=$phase:o=$offset"
        m =
            amplitude == 0 ? LocalPotential1D(offset; id) :
            LocalPotential1D(x -> offset + amplitude * cos(2pi * mode * x / L + phase); id)
        return Problem(
            PeriodicGrid1D(gd["R"], L),
            m,
            get(sd, "occupations", [1.0]);
            mass = get(sd, "mass", 1.0),
        )
    end
    throw(ArgumentError("Unsupported system kind: $kind"))
end
function problem_from_config3d(gd, sd)
    kind = get(sd, "kind", "hydrogen_cluster")
    if kind == "hydrogen_cluster"
        require_keys(sd, ["kind", "shape", "distance", "padding", "xc"], "3D system")
        return hydrogen_cluster(;
            R = get(gd, "R", 7),
            shape = Tuple(get(sd, "shape", [3, 2, 2])),
            distance = get(sd, "distance", 2ANGSTROM_TO_BOHR),
            padding = get(sd, "padding", 10.0),
            cell = get(gd, "cell", nothing),
            xc = Symbol(get(sd, "xc", "pz81")),
        )
    elseif kind == "all_electron"
        require_keys(sd, ["kind", "positions", "charges", "occupations", "mass", "xc"], "3D system")
        m = AllElectronLDA3D(sd["positions"], sd["charges"]; xc = Symbol(get(sd, "xc", "pz81")))
        return Problem(
            BoxGrid3D(gd["R"], gd["cell"]),
            m,
            sd["occupations"];
            mass = get(sd, "mass", 1.0),
        )
    end
    throw(ArgumentError("Unsupported 3D system kind: $kind"))
end

"Parse and validate model settings without constructing fields or orbitals."
function configured_options(settings::AbstractDict)
    require_keys(
        settings,
        [
            "run",
            "grid",
            "system",
            "accuracy",
            "scf",
            "tci",
            "guess",
            "refinement",
            "audit",
            "physics",
            "time",
            "drive",
        ],
        "top-level",
    )
    run = get(settings, "run", Dict())
    require_keys(run, ["kind", "output", "checkpoint", "blas_threads"], "run")
    kind = get(run, "kind", "scf")
    kind in ("scf", "propagation") || throw(ArgumentError("Unknown run kind: $kind"))
    threads = get(run, "blas_threads", 1)
    threads isa Integer && threads >= 1 || throw(ArgumentError("Invalid BLAS thread count"))
    problem = problem_from_config(settings)
    problem.grid isa BoxGrid3D &&
        kind != "scf" &&
        throw(ArgumentError("3D configuration currently supports ground-state SCF"))
    accuracy = options_from_dict(Accuracy, get(settings, "accuracy", Dict()))
    tci = options_from_dict(TCIOptions, get(settings, "tci", Dict()))
    guess = options_from_dict(InitialGuess, get(settings, "guess", Dict()))
    physics = options_from_dict(PhysicsNumerics, get(settings, "physics", Dict()))
    scf = options_from_dict(SCFOptions, get(settings, "scf", Dict()))
    audit = options_from_dict(AuditOptions, get(settings, "audit", Dict()))
    refinement = options_from_dict(RefinementOptions, get(settings, "refinement", Dict()))
    time_options = options_from_dict(TimeOptions, get(settings, "time", Dict()))
    drive_settings = copy(get(settings, "drive", Dict()))
    drive_kind = pop!(drive_settings, "kind", "none")
    drive =
        drive_kind == "none" ?
        (
            isempty(drive_settings) ? NoDrive() :
            throw(ArgumentError("NoDrive accepts no extra parameters"))
        ) :
        drive_kind == "cosine_pulse" ? options_from_dict(CosinePulse, drive_settings) :
        throw(ArgumentError("Unknown drive kind"))
    kind == "scf" &&
        !(drive isa NoDrive) &&
        throw(ArgumentError("SCF has no time-dependent drive; put static fields in the model"))
    foreach(validate, (accuracy, tci, guess, physics, scf, audit, refinement, time_options, drive))
    (;
        problem,
        accuracy,
        tci,
        guess,
        physics,
        scf,
        audit,
        refinement,
        time = time_options,
        drive,
        kind,
        blas_threads = threads,
    )
end

"Validate a TOML input and describe its grid, occupations, and execution settings."
function configuration_plan(path)
    settings = TOML.parsefile(path)
    options = configured_options(settings)
    problem = options.problem
    tasks = options.kind == "scf" ? options.scf.parallel_tasks : options.time.parallel_tasks
    plan = Dict(
        "kind" => options.kind,
        "system" => get(get(settings, "system", Dict()), "kind", "hydrogen_chain"),
        "grid_points" => length(problem.grid),
        "occupied_orbitals" => length(problem.occupations),
        "electrons" => sum(problem.occupations),
        "cell" => problem.grid.L isa Tuple ? collect(problem.grid.L) : problem.grid.L,
        "qtt_sites" => nsites(problem.grid),
        "spatial_dimensions" => problem.grid isa BoxGrid3D ? 3 : 1,
        "julia_threads" => Threads.nthreads(:default),
        "parallel_tasks" => tasks,
        "parallel_workers_for_orbitals" =>
            Q.parallel_workers((; parallel_tasks = tasks), length(problem.occupations)),
        "blas_threads" => options.blas_threads,
        "calculation_started" => false,
        "resolved_field_caps" => option_dict(field_caps(options.accuracy)),
        "dense_orbital_bytes" =>
            big(sizeof(ComplexF64)) * length(problem.grid) * length(problem.occupations),
        "note" => "Dense storage comparison only; not an estimate of peak memory.",
    )
    options.kind == "scf" && (plan["solver"] = string(options.scf.solver))
    plan
end

"Run a TOML experiment. Output directories are fresh; checkpoint states are warm starts."
run_config(path; kwargs...) = run_config(TOML.parsefile(path), path; kwargs...)

function run_config(settings::AbstractDict, path; output = nothing, checkpoint = nothing)
    options = configured_options(settings)
    problem, accuracy, tci, guess, physics =
        options.problem, options.accuracy, options.tci, options.guess, options.physics
    scf, audit, refinement, time_options, drive, kind =
        options.scf, options.audit, options.refinement, options.time, options.drive, options.kind
    run = get(settings, "run", Dict())
    output_directory = output === nothing ? get(run, "output", nothing) : output
    output_directory === nothing && throw(ArgumentError("Specify a fresh output directory"))
    ispath(output_directory) && throw(
        ArgumentError("Run output already exists: $output_directory; choose a fresh directory"),
    )
    checkpoint_path = checkpoint === nothing ? get(run, "checkpoint", nothing) : checkpoint
    checkpoint_path !== nothing &&
        !isabspath(checkpoint_path) &&
        (checkpoint_path = normpath(joinpath(dirname(abspath(path)), checkpoint_path)))
    previous_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(options.blas_threads)
    # Include checkpoint import/refinement in the externally visible configured-run timer.
    start = time_ns()
    try
        initial =
            checkpoint_path === nothing ? nothing :
            load_checkpoint(checkpoint_path; model = problem.model)
        if initial !== nothing && initial.problem.grid.R < problem.grid.R
            tasks = kind == "scf" ? scf.parallel_tasks : time_options.parallel_tasks
            initial = refine_state(
                initial,
                problem;
                accuracy = accuracy,
                tci,
                refinement = refinement,
                parallel_tasks = tasks,
            )
        end
        initial !== nothing &&
            !compatible_problem(initial.problem, problem) &&
            throw(ArgumentError("Checkpoint problem mismatch"))
        result = if kind == "scf"
            pivot_file =
                checkpoint_path === nothing ? nothing :
                joinpath(dirname(checkpoint_path), "cross-state.jls")
            pivot_state =
                tci.reuse_pivots && pivot_file !== nothing && isfile(pivot_file) ? pivot_file :
                nothing
            ground_state(
                problem;
                initial,
                pivot_state,
                accuracy = accuracy,
                scf,
                tci,
                guess,
                physics,
                audit,
                output = output_directory,
            )
        else
            initial === nothing && (
                initial = initial_state(
                    problem,
                    accuracy,
                    recorder(nothing, tci; parallel_tasks = time_options.parallel_tasks),
                    guess,
                )
            )
            propagate(
                initial;
                time = time_options,
                drive,
                accuracy = accuracy,
                tci,
                physics,
                output = output_directory,
            )
        end
        Base.cp(path, joinpath(output_directory, "input.toml"))
        write_json(
            joinpath(output_directory, "workflow.json"),
            (
                elapsed_seconds = (time_ns() - start) / 1e9,
                scope = "configuration through refinement/initialization, solver and output; excludes process startup and postprocessing",
                initial_checkpoint = checkpoint_path,
                refinement = option_dict(refinement),
                guess = option_dict(guess),
            ),
        )
        result
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end
end

# Keep command-line handling separate from configuration and numerical work.
const CLI_USAGE = """
Usage:
  julia --project=. run.jl plan INPUT.toml
  julia --project=. run.jl solve INPUT.toml NEW_OUTPUT
"""

function main(args = ARGS; io = stdout, err = stderr)
    if isempty(args) || args == ["--help"]
        print(io, CLI_USAGE)
        return 0
    end
    try
        if args[1] == "plan"
            length(args) == 2 || throw(ArgumentError("plan needs INPUT.toml"))
            TOML.print(io, configuration_plan(args[2]); sorted = true)
            return 0
        elseif args[1] == "solve"
            length(args) == 3 || throw(ArgumentError("solve needs INPUT.toml and NEW_OUTPUT"))
            result = run_config(args[2]; output = args[3])
            println(io, "Output: ", abspath(args[3]))
            if result isa SCFResult
                println(io, "Energy/iteration stopping reached: ", result.converged)
                println(io, "Independent physical audit passed: ", result.audit.passed)
                return result.converged ? 0 : 2
            end
            println(io, "Propagation steps completed: ", result.completed_steps)
            return 0
        end
        throw(ArgumentError("Unknown command: $(args[1]); use plan or solve"))
    catch exception
        exception isa InterruptException && rethrow()
        println(err, sprint(showerror, exception))
        return 1
    end
end
