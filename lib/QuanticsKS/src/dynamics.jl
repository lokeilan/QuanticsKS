# Split-step propagation: half potential, full kinetic, half potential; save configured outputs.

abstract type AbstractDrive end
struct NoDrive <: AbstractDrive end
"Periodic scalar pulse with a sin² envelope; amplitude is a potential energy in Ha."
Base.@kwdef struct CosinePulse <: AbstractDrive
    amplitude::Float64 = 0.01
    start::Float64 = 0.0
    duration::Float64 = 1.0
    frequency::Float64 = 0.0
    phase::Float64 = 0.0
    spatial_mode::Int = 1
    spatial_phase::Float64 = 0.0
end
struct FunctionDrive{F} <: AbstractDrive
    potential::F
    id::String
end
drive_value(::NoDrive, x, t, L) = 0.0
function drive_value(d::CosinePulse, x, t, L)
    tau = t - d.start
    0 < tau < d.duration || return 0.0
    d.amplitude *
    sinpi(tau / d.duration)^2 *
    cos(d.frequency * tau + d.phase) *
    cos(2pi * d.spatial_mode * x / L + d.spatial_phase)
end
function drive_value(d::FunctionDrive, x, t, L)
    v = d.potential(x, t)
    isreal(v) && isfinite(v) ||
        throw(ArgumentError("Drive must return finite real potential energies"))
    real(v)
end
drive_descriptor(::NoDrive) = Dict("kind" => "none")
drive_descriptor(d::CosinePulse) = merge(Dict("kind" => "cosine_pulse"), option_dict(d))
drive_descriptor(d::FunctionDrive) = Dict("kind" => "function", "id" => d.id)
function validate(d::AbstractDrive)
    if d isa CosinePulse
        d.duration > 0 && all(
            isfinite,
            (d.amplitude, d.start, d.duration, d.frequency, d.phase, d.spatial_phase),
        ) || throw(ArgumentError("Invalid pulse"))
    elseif d isa FunctionDrive
        isempty(d.id) && throw(ArgumentError("Drive id is required"))
    end
    d
end
Base.@kwdef struct TimeOptions
    dt::Float64 = 0.01
    steps::Int = 100
    snapshot_every::Int = 10
    observe_every::Int = 1
    phase_tolerance::Float64 = 1e-10
    phase_cap::Int = 48
    kinetic_tolerance::Float64 = 1e-12
    kinetic_cap::Int = 96
    output_tolerance::Float64 = 1e-12
    output_cap::Int = 48
    norm_drift_limit::Float64 = 1e-5
    gram_drift_limit::Float64 = 1e-5
    seed::Int = 20261031
end
function validate(t::TimeOptions)
    all(
        x -> isfinite(x) && x > 0,
        (
            t.dt,
            t.phase_tolerance,
            t.kinetic_tolerance,
            t.output_tolerance,
            t.norm_drift_limit,
            t.gram_drift_limit,
        ),
    ) || throw(ArgumentError("Invalid propagation tolerances/time step"))
    min(t.steps, t.snapshot_every, t.observe_every, t.phase_cap, t.kinetic_cap, t.output_cap) > 0 ||
        throw(ArgumentError("Invalid propagation counts/caps"))
    t
end
struct TimeResult
    state::OrbitalState
    observations::Vector{Any}
    completed_steps::Int
    elapsed_seconds::Float64
end
function potential_half_step(p, xs, ops, a, rec, t, drive, options)
    n = density(p, xs, a, rec)
    H = effective_hamiltonian(p, n, ops, a, rec; time = t, drive)
    v = Q.evaluate(H.potential, rec)
    [
        begin
            e = Q.evaluate(x, rec)
            Q.cross(
                rec,
                p.grid.R,
                j -> cis(-options.dt * real(v(j)) / 2) * e(j);
                tol = options.phase_tolerance,
                cap = options.phase_cap,
                label = "potential_phase_$i",
            )
        end for (i, x) in enumerate(xs)
    ]
end
function kinetic_step(xs, green, mu, t, rec)
    [
        Q.add(
            Q.apply_mpo(green, x; tol = t.kinetic_tolerance, cap = t.kinetic_cap),
            x;
            a = 2mu,
            b = -1,
            tol = t.output_tolerance,
            cap = t.output_cap,
            rec,
            label = "kinetic_step",
        ) for x in xs
    ]
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
    rec = recorder(dir, tci)
    io = run_io(dir, "observations.jsonl")
    try
        ops = build_operators(p, a, rec, physics)
        mu = 2 / (im * time.dt)
        green = Q.green_mpo(p.grid.R, spacing(p.grid) * sqrt(p.mass), mu)
        state = initial
        observations = Any[]
        S0 = Q.gram(initial.orbitals)
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
            S = Q.gram(xs)
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
