# Observe states and postprocess saved results, sampling orbitals on the requested output grid.

function observe_state(state, ops, accuracy, rec; drive = NoDrive(), overlap = nothing)
    problem = state.problem
    orbitals = state.orbitals
    grid = problem.grid
    S = overlap === nothing ? Q.gram(orbitals; cfg = rec.options) : overlap
    norms = real.(diag(S))
    rho = density(problem, orbitals, accuracy, rec)
    energy = total_energy(problem, orbitals, rho, ops, accuracy, rec; time = state.time, drive)
    charge = sum(problem.occupations .* norms)
    observables = (;
        time = state.time,
        energy,
        electron_number = charge,
        norms,
        gram_defect = norm(S - I),
        max_bond_dimension = Q.orbital_max_rank(orbitals),
        orbital_bytes = sum(Q.storage, orbitals),
    )
    grid isa PeriodicGrid1D || return observables
    moment = if representation(state) == :dense
        spacing(grid) * dot(cis.(-2pi .* (0:(length(grid) - 1)) ./ length(grid)), rho) /
        sum(problem.occupations)
    else
        phase = Q.phase_tt(Q.constant_tt(grid.R), -1)
        spacing(grid) * Q.inner(phase, rho; cfg = rec.options) / sum(problem.occupations)
    end
    merge(observables, (; periodic_cosine = real(moment), periodic_sine = imag(moment)))
end

function observe(
    state::OrbitalState;
    accuracy = Accuracy(),
    tci = TCIOptions(),
    physics = PhysicsNumerics(),
    drive = NoDrive(),
    parallel_tasks = 1,
)
    validate(accuracy)
    validate(drive)
    rec = recorder(nothing, tci; parallel_tasks)
    ops = build_operators(state.problem, accuracy, rec, physics, Val(representation(state)))
    observe_state(state, ops, accuracy, rec; drive)
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
    ops = build_operators(
        first_state.problem,
        accuracy,
        rec,
        physics,
        Val(representation(first_state)),
    )
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
    problem = final.problem
    grid = problem.grid
    dx = volume_element(grid)
    evaluator =
        representation(final) == :dense ? (j -> getindex.(final.orbitals, j + 1)) :
        Q.BlockEvaluator(
            final.orbitals;
            cap = tci.orbital_cache_capacity,
            max_bytes = tci.orbital_cache_max_bytes,
        )
    open(joinpath(out, "density.csv"), "w") do io
        println(io, grid isa BoxGrid3D ? "x,y,z,density" : "x,density")
        count = min(samples, length(grid))
        for index = 0:(count - 1)
            j = round(Int, (big(length(grid)) - 1) * index // (count - 1))
            position = grid_position(grid, j)
            coordinates = position isa Tuple ? join(position, ",") : string(position)
            println(io, "$coordinates,$(sum(problem.occupations.*abs2.(evaluator(j)))/dx)")
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
            println(io, "$i,$k,$(profiles[i,k]),$(2^min(k,nsites(grid)-k)),$cap")
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
