# Split-step propagation: half potential, full kinetic, half potential.

struct TimeResult
    state::OrbitalState
    observations::Vector{Any}
    completed_steps::Int
    elapsed_seconds::Float64
end
function potential_half_step(p, xs, ops, a, rec, t, drive, options)
    n = density(p, xs, a, rec)
    v = local_potential_samples(p, n, ops, a, rec; time = t, drive)
    v = Q.synchronized(v)
    Q.recorded_map(Q.QTT, length(xs), options, rec) do i, localrec
        e = Q.evaluate(xs[i], localrec)
        Q.cross(
            localrec,
            p.grid.R,
            j -> cis(-options.dt * real(v(j)) / 2) * e(j);
            tol = options.phase_tolerance,
            cap = options.phase_cap,
            label = "potential_phase_$i",
        )
    end
end
function kinetic_step(xs, green, mu, t, rec)
    Q.recorded_map(Q.QTT, length(xs), t, rec) do i, localrec
        x = xs[i]
        Q.add(
            Q.apply_mpo(green, x; tol = t.kinetic_tolerance, cap = t.kinetic_cap),
            x;
            a = 2mu,
            b = -1,
            tol = t.output_tolerance,
            cap = t.output_cap,
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
    validate(time)
    validate(drive)
    a = validate(accuracy)
    validate(tci)
    validate(physics)
    t0 = time_ns()
    Random.seed!(time.seed)
    p = initial.problem
    dir = prepare_run(output, "propagation", p; accuracy = a, time, tci, physics)
    dir === nothing || write_json(joinpath(dir, "drive.json"), drive_descriptor(drive))
    rec = recorder(dir, tci; parallel_tasks = time.parallel_tasks)
    io = run_io(dir, "observations.jsonl")
    try
        ops = build_operators(p, a, rec, physics)
        mu = 2 / (im * time.dt)
        green = Q.green_mpo(p.grid.R, spacing(p.grid) * sqrt(p.mass), mu)
        state = initial
        observations = Any[]
        S0 = Q.gram(initial.orbitals; cfg = time)
        norm0 = real.(diag(S0))
        row = observe_state(state, ops, a, rec; drive)
        push!(observations, merge((step = 0,), row))
        Q.event!(io, last(observations))
        dir === nothing || save_checkpoint(joinpath(dir, "snapshots", "step-000000.jls"), state)
        completed = 0
        for k = 1:(time.steps)
            rec.scf = k
            mid = initial.time + (k - 0.5) * time.dt
            xs = potential_half_step(p, state.orbitals, ops, a, rec, mid, drive, time)
            xs = kinetic_step(xs, green, mu, time, rec)
            xs = potential_half_step(p, xs, ops, a, rec, mid, drive, time)
            state = OrbitalState(p, xs; time = initial.time + k * time.dt)
            S = Q.gram(xs; cfg = time)
            nd = maximum(abs.(real.(diag(S)) .- norm0))
            gd = norm(S - S0)
            nd <= time.norm_drift_limit && gd <= time.gram_drift_limit || error(
                "Propagation drift limit exceeded at step $k: norm=$nd Gram=$gd; tighten tolerances or raise caps",
            )
            completed = k
            if k % time.observe_every == 0 || k == time.steps
                row = merge(
                    (step = k, norm_drift = nd, gram_drift = gd),
                    observe_state(state, ops, a, rec; drive),
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
