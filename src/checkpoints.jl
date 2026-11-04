# Save orbital cores and problem metadata; check model and grid identity on load.

const CHECKPOINT_SCHEMA = 1
"Write a versioned data-only Julia checkpoint. Load only trusted local files."
function save_checkpoint(path, state::OrbitalState; metadata = Dict())
    mkpath(dirname(abspath(path)))
    data = Dict(
        "schema_version" => CHECKPOINT_SCHEMA,
        "package_version" => "0.1.0",
        "julia_version" => string(VERSION),
        "problem" => problem_descriptor(state.problem),
        "time" => state.time,
        "cores" => [x.c for x in state.orbitals],
        "metadata" => metadata,
    )
    # Only built-in container/scalar/array types are serialized; no module-owned types or closures.
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, data)
    end
    mv(tmp, path; force = true)
    path
end
function restore_problem(d; model = nothing)
    d["boundary"] == "periodic" && d["units"] == "atomic" ||
        throw(ArgumentError("Unsupported checkpoint conventions"))
    md = d["model"]
    if md["kind"] == "soft_coulomb_lsda_1d"
        m = SoftCoulombLSDA1D(
            md["positions"],
            md["charges"];
            softening = md["softening"],
            coupling = md["coupling"],
            xc = Symbol(md["xc"]),
        )
        model !== nothing && model_descriptor(model) != md && throw(ArgumentError("Model mismatch"))
    elseif md["kind"] == "local_potential_1d"
        model === nothing && throw(
            ArgumentError(
                "Supply the original LocalPotential1D model; executable functions are not serialized",
            ),
        )
        model_descriptor(model) == md || throw(ArgumentError("Potential id mismatch"))
        m = model
    else
        throw(ArgumentError("Unknown model kind"))
    end
    Problem(PeriodicGrid1D(d["R"], d["cell"]), m, d["occupations"]; mass = d["mass"])
end
function load_checkpoint(path; model = nothing, problem = nothing)
    d = deserialize(path)
    d isa AbstractDict && get(d, "schema_version", 0) == CHECKPOINT_SCHEMA ||
        throw(ArgumentError("Unsupported checkpoint schema; expected schema $CHECKPOINT_SCHEMA"))
    p = restore_problem(d["problem"]; model = problem === nothing ? model : problem.model)
    problem !== nothing &&
        !compatible_problem(problem, p) &&
        throw(ArgumentError("Checkpoint problem mismatch"))
    OrbitalState(p, [Q.QTT(c) for c in d["cores"]]; time = d["time"])
end
