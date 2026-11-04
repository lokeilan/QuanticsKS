# Observe states and postprocess saved results, sampling orbitals on the requested output grid.

function observe_state(state, ops, a, rec; drive = NoDrive())
    p = state.problem
    xs = state.orbitals
    g = p.grid
    S = Q.gram(xs; cfg = rec.options)
    norms = real.(diag(S))
    n = density(p, xs, a, rec)
    energy = total_energy(p, xs, n, ops, a, rec; time = state.time, drive)
    charge = sum(p.occupations .* norms)
    q = Q.phase_tt(Q.constant_tt(g.R), -1)
    moment = spacing(g) * Q.inner(q, n; cfg = rec.options) / sum(p.occupations)
    (;
        time = state.time,
        energy,
        electron_number = charge,
        norms,
        gram_defect = norm(S - I),
        periodic_cosine = real(moment),
        periodic_sine = imag(moment),
        max_bond_dimension = maximum(Q.rank.(xs)),
        orbital_bytes = sum(Q.storage, xs),
    )
end
function observe(
    state::OrbitalState;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    physics = PhysicsNumerics(),
    drive = NoDrive(),
    parallel_tasks = 1,
)
    a = validate(accuracy)
    validate(drive)
    rec = recorder(nothing, tci; parallel_tasks)
    ops = build_operators(state.problem, a, rec, physics)
    observe_state(state, ops, a, rec; drive)
end
"Postprocess a completed run without changing snapshots. Custom potential models must be supplied."
function postprocess(
    run;
    model = nothing,
    accuracy = Accuracy(),
    tci = TCIOptions(),
    physics = PhysicsNumerics(),
    drive = nothing,
    samples = 1024,
    orbital_cap = nothing,
    parallel_tasks = 1,
)
    samples >= 2 || throw(ArgumentError("At least two display samples are required"))
    validate(accuracy)
    validate(physics)
    rec = recorder(nothing, tci; parallel_tasks)
    config = JSON3.read(read(joinpath(run, "config.json"), String), Dict{String,Any})
    if drive === nothing
        if isfile(joinpath(run, "drive.json"))
            d = JSON3.read(read(joinpath(run, "drive.json"), String), Dict{String,Any})
            kind = pop!(d, "kind")
            drive =
                kind == "none" ? NoDrive() :
                kind == "cosine_pulse" ? CosinePulse(; (Symbol(k) => v for (k, v) in d)...) :
                throw(ArgumentError("Supply the custom FunctionDrive used for this run"))
        else
            drive = NoDrive()
        end
    end
    validate(drive)
    files = sort(filter(f -> endswith(f, ".jls"), readdir(joinpath(run, "snapshots"); join = true)))
    isempty(files) && (files = [joinpath(run, "checkpoint.jls")])
    out = joinpath(run, "observables")
    mkpath(out)
    first_state = load_checkpoint(first(files); model)
    ops = build_operators(first_state.problem, accuracy, rec, physics)
    rows = Q.recorded_map(Any, length(files), rec.options, rec) do i, localrec
        s = i == 1 ? first_state : load_checkpoint(files[i]; model)
        compatible_problem(first_state.problem, s.problem) ||
            throw(ArgumentError("Snapshot problem mismatch"))
        observe_state(s, ops, accuracy, localrec; drive)
    end
    final = load_checkpoint(joinpath(run, "checkpoint.jls"); model)
    # Include an interrupted run's final state even if it falls between snapshot intervals.
    compatible_problem(first_state.problem, final.problem) ||
        throw(ArgumentError("Final checkpoint problem mismatch"))
    last(rows).time != final.time && push!(rows, observe_state(final, ops, accuracy, rec; drive))
    write_json(joinpath(out, "trajectory.json"), rows)
    p = final.problem
    g = p.grid
    h = spacing(g)
    oracle = Q.BlockOracle(
        final.orbitals;
        cap = tci.orbital_cache_capacity,
        cachebits = tci.environment_cache_bits,
    )
    open(joinpath(out, "density.csv"), "w") do io
        println(io, "x,density")
        for j in unique(round.(Int, range(0, length(g) - 1; length = min(samples, length(g)))))
            println(io, "$(j*h),$(sum(p.occupations.*abs2.(oracle(j)))/h)")
        end
    end
    cap =
        orbital_cap === nothing ?
        Int(
            config["options"][config["kind"] == "propagation" ? "time" : "accuracy"][config["kind"] ==
                                                                                     "propagation" ?
                                                                                     "phase_cap" :
                                                                                     "orbital_cap"],
        ) : orbital_cap
    profiles = bond_profiles(final)
    open(joinpath(out, "bond_profiles.csv"), "w") do io
        println(io, "orbital,bond_index,bond_dimension,algebraic_cap,code_cap")
        for i in axes(profiles, 1), k in axes(profiles, 2)
            println(io, "$i,$k,$(profiles[i,k]),$(2^min(k,g.R-k)),$cap")
        end
    end
    write_json(
        joinpath(out, "postprocess.json"),
        (
            accuracy = option_dict(accuracy),
            tci = option_dict(tci),
            physics = option_dict(physics),
            drive = drive_descriptor(drive),
            samples,
            parallel_tasks,
            profile_order = "MSB to LSB",
            orbital_count = length(final.orbitals),
        ),
    )
    out
end
