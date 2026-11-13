# One orbital propagation workflow with model-specific fields and shared kinetic actions.

struct TimeResult{S}
    state::S
    observations::Vector{Any}
    completed_steps::Int
    elapsed_seconds::Float64
end

function kick(state::OrbitalState{<:Problem,<:AbstractVector}, strength::Real)
    grid = state.problem.grid
    phase = [cis(strength * grid_position(grid, index)) for index = 0:(length(grid) - 1)]
    OrbitalState(state.problem, [phase .* psi for psi in state.orbitals]; time = state.time)
end

function kick(state::OrbitalState{<:Problem,Q.QTT}, strength::Real)
    grid = state.problem.grid
    grid isa BoxGrid3D && throw(ArgumentError("This kick acts on one physical coordinate"))
    orbitals = map(state.orbitals) do psi
        cores = copy.(psi.c)
        cores[1] .*= cis(strength * grid_position(grid, 0))
        for site = 1:nsites(grid)
            cores[site][:, 2, :] .*= cis(strength * spacing(grid) * 2^(site - 1))
        end
        Q.QTT(cores)
    end
    OrbitalState(state.problem, orbitals; time = state.time)
end

function dense_hamiltonian_exponential(H, psi, dt, time)
    result, info = KrylovKit.exponentiate(
        psi -> H.kinetic * psi + H.potential .* psi,
        -im * dt,
        psi;
        ishermitian = true,
        tol = time.exponential_tolerance,
        krylovdim = time.krylov_dimension,
        maxiter = 100,
    )
    info.converged == 1 || error("Hamiltonian exponential did not converge")
    result
end

potential_value(potential::AbstractVector, index) = potential[index + 1]
potential_value(potential, index) = potential(index)

function propagation_potential(
    problem,
    orbitals,
    operators::DenseOperators,
    accuracy,
    rec,
    time,
    drive,
)
    rho = density(problem, orbitals, accuracy, rec)
    effective_hamiltonian(problem, rho, operators, accuracy, rec; time, drive).potential
end
function propagation_potential(problem, orbitals, operators, accuracy, rec, time, drive)
    rho = density(problem, orbitals, accuracy, rec)
    Q.synchronized(local_potential_samples(problem, rho, operators, accuracy, rec; time, drive))
end

function propagate(
    initial::OrbitalState;
    time = TimeOptions(),
    drive = NoDrive(),
    accuracy = Accuracy(),
    tci = TCIOptions(),
    physics = PhysicsNumerics(),
    absorber = nothing,
    output = nothing,
    observer = nothing,
)
    started = time_ns()
    validate(accuracy)
    validate(tci)
    validate(physics)
    validate_propagation(time, drive, absorber)
    problem, grid = initial.problem, initial.problem.grid
    grid isa BoxGrid3D && throw(ArgumentError("3D propagation is not implemented"))
    representation_kind = representation(initial)
    representation_kind == :dense &&
        !(grid isa LineGrid1D) &&
        throw(ArgumentError("Dense propagation requires an explicit LineGrid1D"))
    time.method == :midpoint &&
        representation_kind != :dense &&
        throw(ArgumentError("The midpoint predictor currently requires dense orbitals"))
    Random.seed!(time.seed)
    eta = absorber === nothing ? nothing : Float64.(absorber)
    eta === nothing ||
        (length(eta) == length(grid) && all(value -> isfinite(value) && value >= 0, eta)) ||
        throw(ArgumentError("The CAP must be a nonnegative physical-grid vector"))
    dir = prepare_run(
        output,
        "propagation",
        problem;
        time,
        accuracy,
        tci,
        physics,
        representation = representation_kind,
    )
    rec = recorder(dir, tci; parallel_tasks = time.parallel_tasks)
    operators = build_operators(
        problem,
        accuracy,
        rec,
        physics,
        Val(grid isa LineGrid1D ? :dense : representation_kind),
    )
    if dir !== nothing
        write_json(joinpath(dir, "drive.json"), drive_descriptor(drive))
        write_json(
            joinpath(dir, "absorber.json"),
            (; eta = eta === nothing ? zeros(length(grid)) : eta, renormalize = false),
        )
        if operators isa DenseOperators && operators.xc_table isa SpinXC1D
            write_json(
                joinpath(dir, "xc.json"),
                (;
                    functional = string(problem.model.xc),
                    spin = string(problem.model.spin),
                    libxc_version = operators.xc_table.version,
                ),
            )
        end
    end
    kinetic_actions = Dict{Float64,Any}()
    physical_grid = kinetic_grid(grid)
    kinetic_action(dt) =
        get!(kinetic_actions, dt) do
            representation_kind == :dense ?
            dense_kinetic_propagator(
                physical_grid,
                dt,
                time;
                mass = problem.mass,
                kinetic = operators.kinetic,
            ) : qtt_kinetic_propagator(physical_grid, dt, time, accuracy, rec; mass = problem.mass)
        end
    function kinetic_step(orbitals, dt)
        action = kinetic_action(dt)
        representation_kind == :dense &&
            return [apply_kinetic_step(action, psi, accuracy) for psi in orbitals]
        Q.recorded_map(Q.QTT, length(orbitals), time, rec) do orbital, localrec
            apply_kinetic_step(action, orbitals[orbital], accuracy)
        end
    end
    function potential_step(orbitals, potential, dt, label)
        if representation_kind == :dense
            phase =
                [exp(-im * dt * potential_value(potential, index)) for index = 0:(length(grid) - 1)]
            return [phase .* psi for psi in orbitals]
        end
        Q.recorded_map(Q.QTT, length(orbitals), time, rec) do orbital, localrec
            apply_potential_phase(
                orbitals[orbital],
                index -> potential_value(potential, index),
                dt,
                accuracy,
                localrec;
                label = "$(label)_$orbital",
                support = length(grid),
            )
        end
    end
    potential(orbitals, midpoint) =
        propagation_potential(problem, orbitals, operators, accuracy, rec, midpoint, drive)
    function strang_step(orbitals, dt, midpoint)
        if time.method == :tvt
            orbitals = kinetic_step(orbitals, dt / 2)
            orbitals = potential_step(orbitals, potential(orbitals, midpoint), dt, "potential")
            kinetic_step(orbitals, dt / 2)
        else
            orbitals =
                potential_step(orbitals, potential(orbitals, midpoint), dt / 2, "potential_first")
            orbitals = kinetic_step(orbitals, dt)
            potential_step(orbitals, potential(orbitals, midpoint), dt / 2, "potential_second")
        end
    end
    function midpoint_step(orbitals, dt, midpoint)
        rho = density(problem, orbitals, accuracy, rec)
        H = effective_hamiltonian(
            problem,
            rho,
            operators,
            accuracy,
            rec;
            time = midpoint - dt / 2,
            drive,
        )
        predicted = [dense_hamiltonian_exponential(H, psi, dt / 2, time) for psi in orbitals]
        rho_midpoint = density(problem, predicted, accuracy, rec)
        H_midpoint = effective_hamiltonian(
            problem,
            rho_midpoint,
            operators,
            accuracy,
            rec;
            time = midpoint,
            drive,
        )
        [dense_hamiltonian_exponential(H_midpoint, psi, dt, time) for psi in orbitals]
    end
    initial_overlap = Q.gram(initial.orbitals; cfg = time)
    initial_norms = real.(diag(initial_overlap))
    coefficients = split_step_coefficients(time)
    state, rows, completed = initial, Any[], 0
    io = run_io(dir, "observations.jsonl")
    try
        for step = 0:(time.steps)
            rec.scf = step
            if step > 0
                orbitals = state.orbitals
                eta === nothing ||
                    (orbitals = potential_step(orbitals, -im .* eta, time.dt / 2, "cap_first"))
                midpoint = initial.time + (step - 0.5) * time.dt
                if time.method == :midpoint
                    orbitals = midpoint_step(orbitals, time.dt, midpoint)
                else
                    for coefficient in coefficients
                        orbitals = strang_step(orbitals, coefficient * time.dt, midpoint)
                    end
                end
                eta === nothing ||
                    (orbitals = potential_step(orbitals, -im .* eta, time.dt / 2, "cap_second"))
                state = OrbitalState(problem, orbitals; time = initial.time + step * time.dt)
            end
            overlap = Q.gram(state.orbitals; cfg = time)
            norm_drift = maximum(abs.(real.(diag(overlap)) - initial_norms))
            gram_drift = norm(overlap - initial_overlap)
            if eta === nothing || all(iszero, eta)
                norm_drift <= time.norm_drift_limit && gram_drift <= time.gram_drift_limit || error(
                    "Propagation drift exceeded at step $step: norm=$norm_drift, Gram=$gram_drift",
                )
            end
            if step % time.observe_every == 0 || step == time.steps || step in time.snapshot_steps
                measured = Q.recorded_diagnostic(rec) do diagnostic_rec
                    observe_state(state, operators, accuracy, diagnostic_rec; drive, overlap)
                end
                row = merge((; step, norm_drift, gram_drift), measured)
                push!(rows, row)
                Q.event!(io, row)
                observer === nothing || observer(step, state, row)
            end
            if dir !== nothing
                if step == 0 || step == time.steps || step in time.snapshot_steps
                    save_checkpoint(
                        joinpath(dir, "snapshots", @sprintf("step-%06d.jls", step)),
                        state,
                    )
                end
                if step % max(1, div(time.steps, 100)) == 0
                    write_json(
                        joinpath(dir, "progress.json"),
                        (;
                            step,
                            requested_steps = time.steps,
                            time = state.time,
                            elapsed_seconds = (time_ns() - started) / 1e9,
                        ),
                    )
                end
            end
            completed = step
            dir !== nothing && isfile(joinpath(dir, "STOP")) && break
        end
        elapsed = (time_ns() - started) / 1e9
        if dir !== nothing
            save_checkpoint(joinpath(dir, "checkpoint.jls"), state)
            write_json(
                joinpath(dir, "status.json"),
                (;
                    status = "finished",
                    completed_steps = completed,
                    requested_steps = time.steps,
                    final_time = state.time,
                    elapsed_seconds = elapsed,
                    representation = string(representation_kind),
                    renormalized = false,
                    tci_calls = rec.calls,
                    tci_samples = rec.samples,
                    tci_seconds = rec.tci_time,
                    physical_acceptance = "assessed independently by campaign comparisons",
                ),
            )
        end
        TimeResult(state, rows, completed, elapsed)
    catch exception
        fail_run(dir, exception)
        rethrow()
    finally
        close_run_io(io)
        close(rec)
    end
end
