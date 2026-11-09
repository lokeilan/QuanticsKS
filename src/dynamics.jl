# Split-step propagation: half potential, full kinetic, half potential.

struct TimeResult
    state::OrbitalState
    observations::Vector{Any}
    completed_steps::Int
    elapsed_seconds::Float64
end
function potential_half_step(problem, orbitals, ops, accuracy, rec, t, drive, options)
    rho = density(problem, orbitals, accuracy, rec)
    Veff = local_potential_samples(problem, rho, ops, accuracy, rec; time = t, drive)
    Veff = Q.synchronized(Veff)
    Q.recorded_map(Q.QTT, length(orbitals), options, rec) do i, localrec
        e = Q.evaluate(orbitals[i], localrec)
        Q.cross(
            localrec,
            problem.grid.R,
            j -> cis(-options.dt * real(Veff(j)) / 2) * e(j);
            tol = options.phase_tolerance,
            cap = options.phase_cap,
            label = "potential_phase_$i",
        )
    end
end
function kinetic_step(orbitals, green, mu, options, rec)
    Q.recorded_map(Q.QTT, length(orbitals), options, rec) do i, localrec
        psi = orbitals[i]
        Q.add(
            Q.apply_mpo(green, psi; tol = options.kinetic_tolerance, cap = options.kinetic_cap),
            psi;
            a = 2mu,
            b = -1,
            tol = options.output_tolerance,
            cap = options.output_cap,
            rec = localrec,
            label = "kinetic_step_$i",
        )
    end
end
"Experimental Strang potential / FD2 Cayley kinetic propagator; does not renormalize states."
function propagate(
    initial::OrbitalState;
    time = TimeOptions(),
    drive = NoDrive(),
    accuracy = Accuracy(),
    tci = TCIOptions(),
    physics = PhysicsNumerics(),
    output = nothing,
)
    initial.problem.grid isa BoxGrid3D &&
        throw(ArgumentError("3D real-time propagation is not implemented; use ground_state"))
    validate(time)
    validate(drive)
    validate(accuracy)
    validate(tci)
    validate(physics)
    t0 = time_ns()
    Random.seed!(time.seed)
    problem = initial.problem
    dir = prepare_run(output, "propagation", problem; accuracy = accuracy, time, tci, physics)
    dir === nothing || write_json(joinpath(dir, "drive.json"), drive_descriptor(drive))
    rec = recorder(dir, tci; parallel_tasks = time.parallel_tasks)
    io = run_io(dir, "observations.jsonl")
    try
        ops = build_operators(problem, accuracy, rec, physics)
        mu = 2 / (im * time.dt)
        green = Q.green_mpo(problem.grid.R, spacing(problem.grid) * sqrt(problem.mass), mu)
        state = initial
        observations = Any[]
        S0 = Q.gram(initial.orbitals; cfg = time)
        norm0 = real.(diag(S0))
        row = observe_state(state, ops, accuracy, rec; drive)
        push!(observations, merge((step = 0,), row))
        Q.event!(io, last(observations))
        dir === nothing || save_checkpoint(joinpath(dir, "snapshots", "step-000000.jls"), state)
        completed = 0
        for k = 1:(time.steps)
            rec.scf = k
            mid = initial.time + (k - 0.5) * time.dt
            orbitals =
                potential_half_step(problem, state.orbitals, ops, accuracy, rec, mid, drive, time)
            orbitals = kinetic_step(orbitals, green, mu, time, rec)
            orbitals = potential_half_step(problem, orbitals, ops, accuracy, rec, mid, drive, time)
            state = OrbitalState(problem, orbitals; time = initial.time + k * time.dt)
            S = Q.gram(orbitals; cfg = time)
            nd = maximum(abs.(real.(diag(S)) .- norm0))
            gd = norm(S - S0)
            nd <= time.norm_drift_limit && gd <= time.gram_drift_limit || error(
                "Propagation drift limit exceeded at step $k: norm=$nd Gram=$gd; tighten tolerances or raise caps",
            )
            completed = k
            if k % time.observe_every == 0 || k == time.steps
                row = merge(
                    (step = k, norm_drift = nd, gram_drift = gd),
                    observe_state(state, ops, accuracy, rec; drive),
                )
                push!(observations, row)
                Q.event!(io, row)
            end
            if dir !== nothing && (k % time.snapshot_every == 0 || k == time.steps)
                save_checkpoint(joinpath(dir, "snapshots", @sprintf("step-%06d.jls", k)), state)
            end
            dir !== nothing && isfile(joinpath(dir, "STOP")) && break
        end
        dir === nothing || save_checkpoint(
            joinpath(dir, "checkpoint.jls"),
            state;
            metadata = Dict("completed_steps" => completed, "propagator" => "strang_fd2_cayley"),
        )
        elapsed = (time_ns() - t0) / 1e9
        dir === nothing || write_json(
            joinpath(dir, "status.json"),
            (
                status = "finished",
                completed_steps = completed,
                requested_steps = time.steps,
                elapsed_seconds = elapsed,
                final_time = state.time,
            ),
        )
        TimeResult(state, observations, completed, elapsed)
    catch e
        fail_run(dir, e)
        rethrow()
    finally
        close_run_io(io)
        close(rec)
    end
end
