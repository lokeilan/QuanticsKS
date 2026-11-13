# Two physical 1D electrons in configuration space, distinct from KS density feedback.

struct CorrelatedProblem
    grid::LineGrid1D
    positions::Vector{Float64}
    charges::Vector{Float64}
    nuclear_constant::Float64
    function CorrelatedProblem(
        grid,
        positions,
        charges = ones(length(positions));
        nuclear_constant = 0.0,
    )
        !isempty(positions) &&
            length(positions) == length(charges) &&
            all(isfinite, positions) &&
            all(charge -> isfinite(charge) && charge > 0, charges) &&
            isfinite(nuclear_constant) || throw(ArgumentError("Invalid correlated nuclei"))
        new(grid, Float64.(positions), Float64.(charges), nuclear_constant)
    end
end

"Configuration amplitude A_ij=h*Psi(x_i,x_j); its Euclidean norm is one."
struct CorrelatedState{A}
    problem::CorrelatedProblem
    amplitude::A
    time::Float64
    function CorrelatedState(problem, amplitude::AbstractMatrix; time = 0.0)
        size(amplitude) == (length(problem.grid), length(problem.grid)) &&
            all(isfinite, amplitude) &&
            norm(amplitude) > 0 &&
            isfinite(time) || throw(ArgumentError("Invalid correlated amplitude"))
        new{Matrix{ComplexF64}}(problem, ComplexF64.(amplitude), time)
    end
    function CorrelatedState(problem, amplitude::Q.QTT; time = 0.0)
        grid = problem.grid
        ispow2(grid.n) &&
            length(amplitude) == 2grid.R &&
            isfinite(time) &&
            Q.ttnorm(amplitude) > 0 ||
            throw(ArgumentError("Correlated QTT requires two complete binary coordinate grids"))
        new{Q.QTT}(problem, amplitude, time)
    end
end
representation(::CorrelatedState{<:AbstractMatrix}) = :dense
representation(::CorrelatedState{Q.QTT}) = :qtt
correlated_amplitude(state::CorrelatedState{<:AbstractMatrix}) = state.amplitude
correlated_amplitude(state::CorrelatedState{Q.QTT}) =
    reshape(Q.full_values(state.amplitude), state.problem.grid.n, :)

function problem_descriptor(problem::CorrelatedProblem)
    grid = problem.grid
    Dict(
        "kind" => "correlated_two_electron_1d",
        "physical_dimensions" => 1,
        "configuration_dimensions" => 2,
        "n" => grid.n,
        "spacing" => grid.h,
        "origin" => grid.origin,
        "cell" => grid.L,
        "boundary" => string(grid.boundary),
        "fd_order" => grid.fd_order,
        "units" => "atomic",
        "positions" => problem.positions,
        "charges" => problem.charges,
        "nuclear_constant" => problem.nuclear_constant,
        "amplitude_convention" => "h*Psi",
    )
end

"Compress a supplied configuration-space amplitude without changing its normalization."
function compress_state(
    state::CorrelatedState{<:AbstractMatrix};
    tolerance = 1e-12,
    cap = 128,
    tci = TCIOptions(),
)
    grid = state.problem.grid
    ispow2(grid.n) || throw(ArgumentError("Correlated QTT requires a binary grid"))
    rec = recorder(nothing, tci)
    try
        values = vec(state.amplitude)
        amplitude = Q.cross(
            rec,
            2grid.R,
            index -> values[index + 1];
            tol = tolerance,
            cap,
            label = "supplied_correlated_amplitude",
        )
        CorrelatedState(state.problem, amplitude; time = state.time)
    finally
        close(rec)
    end
end

function project_exchange_inversion(amplitude, inversion_parity)
    exchange_even = (amplitude + transpose(amplitude)) / 2
    (exchange_even + inversion_parity .* reverse(exchange_even; dims = (1, 2))) / 2
end

"Lowest exchange-even eigenstate in the declared spatial inversion sector."
function correlated_eigenstate(
    problem::CorrelatedProblem;
    inversion_parity = 1,
    tolerance = 1e-10,
    max_iterations = 800,
    initial_box = 96.0,
)
    inversion_parity in (-1, 1) || throw(ArgumentError("Inversion parity must be +1 or -1"))
    grid = problem.grid
    physical_nodes = min(grid.n, max(16, 2ceil(Int, initial_box / (2grid.h))))
    offset = div(grid.n - physical_nodes, 2)
    initial_grid = LineGrid1D(
        physical_nodes,
        grid.h;
        origin = grid.origin + offset * grid.h,
        boundary = physical_nodes == grid.n ? grid.boundary : :open,
        fd_order = grid.fd_order,
        cell = physical_nodes * grid.h,
    )
    initial_problem = CorrelatedProblem(
        initial_grid,
        problem.positions,
        problem.charges;
        nuclear_constant = problem.nuclear_constant,
    )
    operators = correlated_operators(initial_problem, Val(:dense))
    gaussian = exp.(-operators.coordinates .^ 2 / 8)
    amplitude = gaussian * transpose(gaussian)
    inversion_parity == -1 &&
        (amplitude .*= operators.coordinates .+ transpose(operators.coordinates))
    amplitude = normalize(project_exchange_inversion(amplitude, inversion_parity))
    kinetic_energies = Q.kinetic_eigenvalues(physical_nodes, grid.h, grid.fd_order)
    denominator = kinetic_energies .+ transpose(kinetic_energies) .+ 1.0
    action(amplitude) =
        project_exchange_inversion(correlated_action(operators, amplitude), inversion_parity)
    precondition(residual) =
        project_exchange_inversion(real.(ifft(fft(residual) ./ denominator)), inversion_parity)
    function progress(iteration, energy, residual)
        if iteration % 25 == 0
            println("Eigenstate iteration ", iteration, " energy ", energy, " residual ", residual)
            flush(stdout)
        end
    end
    solved = Q.preconditioned_eigenstate(
        action,
        precondition,
        amplitude;
        tolerance = tolerance / 3,
        max_iterations,
        callback = progress,
    )
    local_amplitude = ComplexF64.(solved.amplitude)
    pivot = argmax(abs.(local_amplitude))
    local_amplitude .*= conj(local_amplitude[pivot]) / abs(local_amplitude[pivot])
    amplitude = zeros(ComplexF64, grid.n, grid.n)
    amplitude[(offset + 1):(offset + physical_nodes), (offset + 1):(offset + physical_nodes)] .=
        local_amplitude
    state = CorrelatedState(problem, amplitude)
    # The requested tolerance applies to the actual full physical-grid Hamiltonian.
    full_operators =
        physical_nodes == grid.n ? operators : correlated_operators(problem, Val(:dense))
    H_amplitude = correlated_action(full_operators, amplitude)
    electronic_energy = real(dot(amplitude, H_amplitude))
    residual = norm(H_amplitude - electronic_energy .* amplitude)
    residual <= tolerance || error(
        "Full-grid eigenstate residual $residual exceeds $tolerance; enlarge initial_box or tighten the solve",
    )
    (;
        state,
        electronic_energy,
        residual,
        inversion_parity,
        exchange_defect = norm(amplitude - transpose(amplitude)),
        inversion_defect = norm(amplitude - inversion_parity .* reverse(amplitude; dims = (1, 2))),
        preparation = "independent configuration-space eigenstate; full-grid residual checked",
        eigensolver_iterations = solved.iterations,
        initial_box = physical_nodes * grid.h,
    )
end

function kick(state::CorrelatedState{<:AbstractMatrix}, strength::Real)
    grid = state.problem.grid
    phase = [cis(strength * grid_position(grid, index)) for index = 0:(grid.n - 1)]
    CorrelatedState(state.problem, state.amplitude .* (phase * transpose(phase)); time = state.time)
end
function kick(state::CorrelatedState{Q.QTT}, strength::Real)
    grid = state.problem.grid
    cores = copy.(state.amplitude.c)
    cores[1] .*= cis(2strength * grid.origin)
    for site = 1:(2grid.R)
        cores[site][:, 2, :] .*= cis(strength * grid.h * 2^mod(site - 1, grid.R))
    end
    CorrelatedState(state.problem, Q.QTT(cores); time = state.time)
end

function save_correlated_checkpoint(path, state; metadata = (;))
    problem, grid = state.problem, state.problem.grid
    mkpath(dirname(path))
    data = Dict(
        "kind" => "correlated_two_electron_1d",
        "schema_version" => 1,
        "amplitude" => state.amplitude isa Q.QTT ? state.amplitude.c : state.amplitude,
        "representation" =>
            state.amplitude isa Q.QTT ? "qtt_grouped_coordinate_bits" : "dense_configuration_space",
        "time" => state.time,
        "positions" => problem.positions,
        "charges" => problem.charges,
        "nuclear_constant" => problem.nuclear_constant,
        "n" => grid.n,
        "spacing" => grid.h,
        "origin" => grid.origin,
        "cell" => grid.L,
        "boundary" => string(grid.boundary),
        "fd_order" => grid.fd_order,
        "metadata" => metadata,
    )
    open(path * ".tmp", "w") do io
        serialize(io, data)
    end
    mv(path * ".tmp", path; force = true)
end
function load_correlated_checkpoint(path)
    data = deserialize(path)
    data["kind"] == "correlated_two_electron_1d" ||
        throw(ArgumentError("Not a correlated checkpoint"))
    grid = LineGrid1D(
        data["n"],
        data["spacing"];
        origin = data["origin"],
        boundary = Symbol(data["boundary"]),
        fd_order = data["fd_order"],
        cell = data["cell"],
    )
    problem = CorrelatedProblem(
        grid,
        data["positions"],
        data["charges"];
        nuclear_constant = data["nuclear_constant"],
    )
    amplitude =
        data["representation"] == "qtt_grouped_coordinate_bits" ? Q.QTT(data["amplitude"]) :
        data["amplitude"]
    CorrelatedState(problem, amplitude; time = data["time"])
end

function correlated_propagate(
    initial::CorrelatedState;
    time = TimeOptions(method = :yoshida4, kinetic_exponential = :pade4, dt = 0.025),
    accuracy = Accuracy(),
    tci = TCIOptions(),
    output = nothing,
    observer = nothing,
)
    started = time_ns()
    validate(accuracy)
    validate(tci)
    validate_propagation(time, NoDrive(), nothing)
    time.method in (:vtv, :yoshida4) ||
        throw(ArgumentError("Correlated propagation uses vtv or autonomous yoshida4"))
    Random.seed!(time.seed)
    problem, grid = initial.problem, initial.problem.grid
    representation_kind = representation(initial)
    dir = prepare_run(
        output,
        "correlated_propagation",
        problem;
        time,
        accuracy,
        tci,
        representation = representation_kind,
    )
    rec = recorder(dir, tci; parallel_tasks = time.parallel_tasks)
    operators = correlated_operators(problem, Val(representation_kind))
    dir === nothing || save_correlated_checkpoint(joinpath(dir, "initial.jls"), initial)
    kinetic_actions = Dict{Float64,Any}()
    function kinetic_action(dt)
        get!(kinetic_actions, dt) do
            if representation_kind == :dense
                action = dense_kinetic_propagator(grid, dt, time; kinetic = operators.kinetic)
                amplitude -> permutedims(action(permutedims(action(amplitude))))
            else
                base = qtt_kinetic_propagator(grid, dt, time, accuracy, rec)
                axes = vcat(fill(1, grid.R), fill(2, grid.R))
                first_axis = lift_kinetic_propagator(base, 1, axes)
                second_axis = lift_kinetic_propagator(base, 2, axes)
                amplitude -> apply_kinetic_step(
                    second_axis,
                    apply_kinetic_step(first_axis, amplitude, accuracy),
                    accuracy,
                )
            end
        end
    end
    phase_actions = Dict{Float64,Any}()
    function phase_step(amplitude, dt, label)
        if representation_kind == :dense
            phase = get!(phase_actions, dt) do
                cis.(-dt .* operators.potential)
            end
            return phase .* amplitude
        end
        apply_potential_phase(amplitude, operators.potential, dt, accuracy, rec; label)
    end
    coefficients = split_step_coefficients(time)
    amplitude =
        representation_kind == :dense ? copy(initial.amplitude) : Q.QTT(copy.(initial.amplitude.c))
    state, rows, completed = initial, Any[], 0
    initial_norm = representation_kind == :dense ? norm(amplitude)^2 : Q.ttnorm(amplitude)^2
    io = run_io(dir, "observations.jsonl")
    try
        for step = 0:(time.steps)
            rec.scf = step
            if step > 0
                for coefficient in coefficients
                    dt = coefficient * time.dt
                    amplitude = phase_step(amplitude, dt / 2, "potential_first_$coefficient")
                    amplitude = kinetic_action(dt)(amplitude)
                    amplitude = phase_step(amplitude, dt / 2, "potential_second_$coefficient")
                end
                state = CorrelatedState(problem, amplitude; time = initial.time + step * time.dt)
            end
            norm2 = representation_kind == :dense ? sum(abs2, amplitude) : Q.ttnorm(amplitude)^2
            abs(norm2 - initial_norm) <= time.norm_drift_limit ||
                error("Correlated norm drift exceeded at step $step")
            if step % time.observe_every == 0 || step == time.steps || step in time.snapshot_steps
                measured = Q.recorded_diagnostic(rec) do diagnostic_rec
                    correlated_observe(state, operators; accuracy, rec = diagnostic_rec)
                end
                row = (;
                    step,
                    max_bond_dimension = representation_kind == :qtt ? Q.rank(amplitude) : nothing,
                    (name => value for (name, value) in pairs(measured) if name != :rho)...,
                )
                push!(rows, row)
                Q.event!(io, row)
                observer === nothing || observer(step, state, measured)
                if dir !== nothing &&
                   (step == 0 || step == time.steps || step in time.snapshot_steps)
                    write_json(
                        joinpath(dir, @sprintf("density-%06d.json", step)),
                        (; time = state.time, rho = measured.rho),
                    )
                end
            end
            if dir !== nothing && step % max(1, div(time.steps, 100)) == 0
                write_json(
                    joinpath(dir, "progress.json"),
                    (;
                        step,
                        requested_steps = time.steps,
                        elapsed_seconds = (time_ns() - started) / 1e9,
                    ),
                )
            end
            completed = step
            dir !== nothing && isfile(joinpath(dir, "STOP")) && break
        end
        elapsed = (time_ns() - started) / 1e9
        if dir !== nothing
            save_correlated_checkpoint(joinpath(dir, "checkpoint.jls"), state)
            write_json(
                joinpath(dir, "status.json"),
                (;
                    status = "finished",
                    completed_steps = completed,
                    requested_steps = time.steps,
                    final_time = state.time,
                    elapsed_seconds = elapsed,
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
