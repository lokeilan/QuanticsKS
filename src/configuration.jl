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
    require_keys(gd, ["R", "cell"], "grid")
    sd = get(d, "system", Dict())
    kind = get(sd, "kind", "hydrogen_chain")
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
"Parse and validate one-dimensional settings without constructing fields or orbitals."
function configured_options(d::AbstractDict)
    require_keys(
        d,
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
    run = get(d, "run", Dict())
    require_keys(run, ["kind", "output", "checkpoint", "blas_threads"], "run")
    kind = get(run, "kind", "scf")
    kind in ("scf", "propagation") || throw(ArgumentError("Unknown run kind: $kind"))
    threads = get(run, "blas_threads", 1)
    threads isa Integer && threads >= 1 || throw(ArgumentError("Invalid BLAS thread count"))
    p = problem_from_config(d)
    a = options_from_dict(Accuracy, get(d, "accuracy", Dict()))
    tci = options_from_dict(TCIOptions, get(d, "tci", Dict()))
    guess = options_from_dict(InitialGuess, get(d, "guess", Dict()))
    physics = options_from_dict(PhysicsNumerics, get(d, "physics", Dict()))
    scf = options_from_dict(SCFOptions, get(d, "scf", Dict()))
    audit = options_from_dict(AuditOptions, get(d, "audit", Dict()))
    ref = options_from_dict(RefinementOptions, get(d, "refinement", Dict()))
    to = options_from_dict(TimeOptions, get(d, "time", Dict()))
    dd = copy(get(d, "drive", Dict()))
    dk = pop!(dd, "kind", "none")
    drive =
        dk == "none" ?
        (isempty(dd) ? NoDrive() : throw(ArgumentError("NoDrive accepts no extra parameters"))) :
        dk == "cosine_pulse" ? options_from_dict(CosinePulse, dd) :
        throw(ArgumentError("Unknown drive kind"))
    kind == "scf" &&
        !(drive isa NoDrive) &&
        throw(ArgumentError("SCF has no time-dependent drive; put static fields in the model"))
    foreach(validate, (a, tci, guess, physics, scf, audit, ref, to, drive))
    (;
        problem = p,
        accuracy = a,
        tci,
        guess,
        physics,
        scf,
        audit,
        refinement = ref,
        time = to,
        drive,
        kind,
        blas_threads = threads,
    )
end

"Validate a TOML input and describe its grid, occupations, and execution settings."
function configuration_plan(path)
    d = TOML.parsefile(path)
    c = configured_options(d)
    p = c.problem
    tasks = c.kind == "scf" ? c.scf.parallel_tasks : c.time.parallel_tasks
    Dict(
        "kind" => c.kind,
        "system" => get(get(d, "system", Dict()), "kind", "hydrogen_chain"),
        "grid_points" => length(p.grid),
        "occupied_orbitals" => length(p.occupations),
        "electrons" => sum(p.occupations),
        "cell" => p.grid.L,
        "julia_threads" => Threads.nthreads(:default),
        "parallel_tasks" => tasks,
        "parallel_workers_for_orbitals" =>
            Q.parallel_workers((; parallel_tasks = tasks), length(p.occupations)),
        "blas_threads" => c.blas_threads,
        "calculation_started" => false,
        "dense_orbital_bytes" => sizeof(ComplexF64) * length(p.grid) * length(p.occupations),
        "note" => "Dense storage comparison only; not an estimate of peak memory.",
    )
end

"Run a TOML experiment. Output directories are fresh; checkpoint states are warm starts."
run_config(path; kwargs...) = run_config(TOML.parsefile(path), path; kwargs...)

function run_config(d::AbstractDict, path; output = nothing, checkpoint = nothing)
    c = configured_options(d)
    p, a, tci, guess, physics = c.problem, c.accuracy, c.tci, c.guess, c.physics
    scf, audit, ref, to, drive, kind = c.scf, c.audit, c.refinement, c.time, c.drive, c.kind
    run = get(d, "run", Dict())
    out = output === nothing ? get(run, "output", nothing) : output
    out === nothing && throw(ArgumentError("Specify a fresh output directory"))
    ispath(out) && throw(ArgumentError("Run output already exists: $out; choose a fresh directory"))
    cp = checkpoint === nothing ? get(run, "checkpoint", nothing) : checkpoint
    cp !== nothing && !isabspath(cp) && (cp = normpath(joinpath(dirname(abspath(path)), cp)))
    oldthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(c.blas_threads)
    # Include checkpoint import/refinement in the externally visible configured-run timer.
    start = time_ns()
    try
        initial = cp === nothing ? nothing : load_checkpoint(cp; model = p.model)
        if initial !== nothing && initial.problem.grid.R < p.grid.R
            tasks = kind == "scf" ? scf.parallel_tasks : to.parallel_tasks
            initial = refine_state(
                initial,
                p;
                accuracy = a,
                tci,
                refinement = ref,
                parallel_tasks = tasks,
            )
        end
        initial !== nothing &&
            !compatible_problem(initial.problem, p) &&
            throw(ArgumentError("Checkpoint problem mismatch"))
        result = if kind == "scf"
            pivot_file = cp === nothing ? nothing : joinpath(dirname(cp), "cross-state.jls")
            pivot_state =
                tci.reuse_pivots && pivot_file !== nothing && isfile(pivot_file) ? pivot_file :
                nothing
            ground_state(
                p;
                initial,
                pivot_state,
                accuracy = a,
                scf,
                tci,
                guess,
                physics,
                audit,
                output = out,
            )
        else
            initial === nothing && (
                initial = initial_state(
                    p,
                    a,
                    recorder(nothing, tci; parallel_tasks = to.parallel_tasks),
                    guess,
                )
            )
            propagate(initial; time = to, drive, accuracy = a, tci, physics, output = out)
        end
        Base.cp(path, joinpath(out, "input.toml"))
        write_json(
            joinpath(out, "workflow.json"),
            (
                elapsed_seconds = (time_ns() - start) / 1e9,
                scope = "configuration through refinement/initialization, solver and output; excludes process startup and postprocessing",
                initial_checkpoint = cp,
                refinement = option_dict(ref),
                guess = option_dict(guess),
            ),
        )
        result
    finally
        BLAS.set_num_threads(oldthreads)
    end
end
