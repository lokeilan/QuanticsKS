# Observe states and postprocess saved results, sampling orbitals on the requested output grid.

function correlated_observe(
    state::CorrelatedState{<:AbstractMatrix},
    operators = correlated_operators(state.problem, Val(:dense));
    accuracy = Accuracy(),
    rec = nothing,
    channel_radius = 8.0,
    edge_fraction = 0.8,
)
    grid, amplitude = state.problem.grid, state.amplitude
    probability = abs2.(amplitude)
    marginal = vec(sum(probability; dims = 2)) + vec(sum(probability; dims = 1))
    norm2 = sum(probability)
    kinetic =
        real(dot(amplitude, operators.kinetic * amplitude)) +
        real(dot(permutedims(amplitude), operators.kinetic * permutedims(amplitude)))
    electronic_energy = kinetic + sum(operators.potential .* probability)
    bound = abs.(operators.coordinates) .<= channel_radius
    interior = abs.(operators.coordinates) .<= edge_fraction * grid.L / 2
    bound_probability = sum(probability[bound, bound])
    double_probability = sum(probability[.!bound, .!bound])
    (;
        time = state.time,
        norm2,
        electron_number = 2norm2,
        electronic_dipole = -dot(operators.coordinates, marginal),
        mean_square_position = dot(operators.coordinates .^ 2, marginal) / 2,
        electronic_energy,
        energy = electronic_energy + state.problem.nuclear_constant * norm2,
        interaction = sum(operators.interaction .* probability),
        exchange_defect = norm(amplitude - transpose(amplitude)),
        edge = max(0.0, norm2 - sum(probability[interior, interior])),
        bound_probability,
        single_probability = norm2 - bound_probability - double_probability,
        double_probability,
        rho = marginal / grid.h,
    )
end

"Contract both one-coordinate marginals from the existing endpoint tables, without an N^2 array."
function correlated_marginals(state::CorrelatedState{Q.QTT})
    grid, amplitude = state.problem.grid, state.amplitude
    left = transpose(Q.endpoint_table(amplitude, grid.R, true))
    right = Q.endpoint_table(amplitude, grid.R, false)
    left_gram, right_gram = left' * left, right * right'
    marginal_first = max.(0.0, real.(vec(sum((left * right_gram) .* conj.(left); dims = 2))))
    marginal_second = max.(0.0, real.(vec(sum(conj.(right) .* (left_gram * right); dims = 1))))
    (; left, right, marginal_first, marginal = marginal_first + marginal_second)
end

"Evaluate the antisymmetric norm through thin QR factors, avoiding norm cancellation."
function correlated_exchange_defect(factors)
    left = hcat(factors.left, transpose(factors.right))
    right = vcat(factors.right, -transpose(factors.left))
    norm(qr(left).R * qr(right').R')
end

function correlated_weighted_expectation(amplitude, potential, accuracy, rec, label; pivots)
    evaluator = Q.Evaluator(amplitude)
    weighted = Q.cross(
        rec,
        length(amplitude),
        index -> potential(index) * evaluator(index);
        tol = accuracy.field_tolerance,
        cap = accuracy.work_cap,
        label,
        pivots,
    )
    real(Q.inner(amplitude, weighted))
end

function correlated_observe(
    state::CorrelatedState{Q.QTT},
    operators = correlated_operators(state.problem, Val(:qtt));
    accuracy = Accuracy(),
    rec = nothing,
    channel_radius = 8.0,
    edge_fraction = 0.8,
)
    own_recorder = rec === nothing
    own_recorder && (rec = recorder(nothing, TCIOptions()))
    try
        grid, amplitude = state.problem.grid, state.amplitude
        factors = correlated_marginals(state)
        norm2 = real(Q.inner(amplitude, amplitude))
        kinetic =
            operators.kinetic_diagonal * norm2 + sum(
                term.weight * real(Q.inner_mpo(amplitude, term.operator, amplitude)) for
                term in operators.kinetic_terms
            )
        interaction = correlated_weighted_expectation(
            amplitude,
            operators.interaction,
            accuracy,
            rec,
            "interaction_energy";
            pivots = correlated_pivots(factors),
        )
        electronic_energy = kinetic + dot(operators.nuclear, factors.marginal) + interaction
        function probability(first, second)
            left = factors.left[first, :]
            right = factors.right[:, second]
            real(tr((left' * left) * (right * right')))
        end
        bound = abs.(operators.coordinates) .<= channel_radius
        interior = abs.(operators.coordinates) .<= edge_fraction * grid.L / 2
        bound_probability = probability(bound, bound)
        double_probability = probability(.!bound, .!bound)
        exchange_defect = correlated_exchange_defect(factors)
        (;
            time = state.time,
            norm2,
            electron_number = 2norm2,
            electronic_dipole = -dot(operators.coordinates, factors.marginal),
            mean_square_position = dot(operators.coordinates .^ 2, factors.marginal) / 2,
            electronic_energy,
            energy = electronic_energy + state.problem.nuclear_constant * norm2,
            interaction,
            exchange_defect,
            edge = max(0.0, norm2 - probability(interior, interior)),
            bound_probability,
            single_probability = norm2 - bound_probability - double_probability,
            double_probability,
            rho = factors.marginal / grid.h,
        )
    finally
        own_recorder && close(rec)
    end
end

function observe_state(state, ops, accuracy, rec; drive = NoDrive(), overlap = nothing)
    problem = state.problem
    orbitals = state.orbitals
    grid = problem.grid
    S = overlap === nothing ? Q.gram(orbitals; cfg = rec.options) : overlap
    norms = real.(diag(S))
    physical_orbitals =
        grid isa LineGrid1D ? [line_orbital_values(psi, grid) for psi in orbitals] : orbitals
    rho = density(problem, physical_orbitals, accuracy, rec)
    energy =
        total_energy(problem, physical_orbitals, rho, ops, accuracy, rec; time = state.time, drive)
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
    if grid isa LineGrid1D
        x = [grid_position(grid, j) for j = 0:(grid.n - 1)]
        return merge(
            observables,
            (;
                electronic_dipole = -grid.h * dot(x, rho),
                mean_square_position = grid.h * dot(x .^ 2, rho) / sum(problem.occupations),
                physical_electron_number = grid.h * sum(rho),
                padding_electron_number = charge - grid.h * sum(rho),
            ),
        )
    end
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
    cap = orbital_cap === nothing ? Int(config["options"]["accuracy"]["orbital_cap"]) : orbital_cap
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
