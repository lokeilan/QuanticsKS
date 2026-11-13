# SCF density feedback, output Ritz rotation, and independent physical audit.

struct SCFResult
    state::OrbitalState
    history::Vector{Any}
    accepted::Bool
    stop_reason::Symbol
    audit::NamedTuple
    elapsed_seconds::Float64
end

"Shared density-feedback SCF; history starts fresh for QTT and dense orbitals."
function ground_state(
    problem::Problem;
    initial = nothing,
    representation::Symbol = initial === nothing ? :qtt : QuanticsKS.representation(initial),
    pivot_state = nothing,
    accuracy = Accuracy(),
    scf = SCFOptions(),
    solver::Symbol = scf.solver,
    tci = TCIOptions(),
    guess = InitialGuess(),
    physics = PhysicsNumerics(),
    audit = AuditOptions(),
    refinement = RefinementOptions(),
    output = nothing,
)
    scf = solver == scf.solver ? scf : SCFOptions(; merge(option_tuple(scf), (; solver))...)
    representation in (:qtt, :dense) ||
        throw(ArgumentError("Unknown orbital representation: $representation"))
    representation == :dense &&
        solver != :lrrap &&
        throw(ArgumentError("Dense SCF currently supports solver=:lrrap only"))
    initial === nothing ||
        QuanticsKS.representation(initial) == representation ||
        throw(ArgumentError("Initial orbital representation mismatch"))
    representation == :dense &&
        pivot_state !== nothing &&
        throw(ArgumentError("TCI pivot state requires QTT orbitals"))
    validate(accuracy)
    validate(scf)
    validate(tci)
    validate(physics)
    validate(guess)
    if initial === nothing && scf.initial_spacing !== nothing
        return ground_state_refined(
            problem;
            representation,
            accuracy,
            scf,
            solver,
            tci,
            guess,
            physics,
            audit,
            refinement,
            output,
        )
    end
    all(==(first(problem.occupations)), problem.occupations) ||
        throw(ArgumentError("Ground-state SCF requires equal occupations"))
    occupied_count = length(problem.occupations)
    orbital_count = occupied_count + scf.extra_orbitals
    orbital_count <= length(problem.grid) ||
        throw(ArgumentError("Orbital block exceeds grid dimension"))
    initial !== nothing &&
        !compatible_problem(initial.problem, problem) &&
        throw(
            ArgumentError("Initial state mismatch; call refine_state explicitly for a finer grid"),
        )
    t0 = time_ns()
    Random.seed!(scf.seed)
    dir = prepare_run(
        output,
        "scf",
        problem;
        accuracy = accuracy,
        scf,
        tci,
        guess,
        physics,
        audit,
        representation,
    )
    rec = recorder(dir, tci; parallel_tasks = scf.parallel_tasks)
    sio = run_io(dir, "scf.jsonl")
    solver_io = run_io(dir, solver == :lrrap ? "lrrap.jsonl" : "eigensolver.jsonl")
    try
        pivot_state === nothing || Q.load_cross_state!(rec, pivot_state)
        base_accuracy = accuracy
        precision_factor = accuracy.adaptive && initial === nothing ? 100.0 : 1.0
        accuracy = iteration_accuracy(problem, base_accuracy, scf, precision_factor)
        cfg = solver_config(problem, accuracy, scf)
        ops = Q.timed(rec, "initial_operators") do
            build_operators(problem, accuracy, rec, physics, Val(representation))
        end
        green = Q.timed(rec, "initial_green_operator") do
            solver == :block_lanczos ? nothing : green_operator(problem, accuracy, scf, physics, ops)
        end
        orbitals = Q.timed(rec, "initial_orbitals") do
            if initial === nothing
                initial_orbitals(
                    problem,
                    accuracy,
                    rec,
                    guess,
                    Val(representation);
                    count = orbital_count,
                )
            elseif scf.extra_orbitals == 0
                copy(initial.orbitals)
            else
                candidates = initial_orbitals(
                    problem,
                    accuracy,
                    rec,
                    guess,
                    Val(representation);
                    count = orbital_count,
                )
                # Project the starting modes away from the supplied occupied space.
                extra = Q.extend_subspace(
                    initial.orbitals,
                    candidates,
                    scf.extra_orbitals,
                    merge(cfg, (; work_cap = accuracy.orbital_cap)),
                    rec;
                    tol = accuracy.orthogonalization_tolerance,
                )
                length(extra) == scf.extra_orbitals ||
                    error("Initial modes do not span enough extra orbitals")
                vcat(initial.orbitals, extra)
            end
        end
        state = OrbitalState(problem, orbitals[1:occupied_count])
        dir === nothing || save_checkpoint(
            joinpath(dir, "initial.jls"),
            state;
            metadata = Dict("source" => initial === nothing ? "initial_guess" : "provided_state"),
        )
        previous_density = nothing
        preconditioner = density_preconditioner(problem, ops, base_accuracy, scf, rec)
        output_density = nothing
        search_directions = nothing
        previous_energy = Inf
        previous_density_residual = Inf
        best_density_residual = Inf
        mixing_parameter = scf.mixing
        density_history = representation == :qtt ? Q.QTT[] : Vector{Float64}[]
        mixing_residuals = representation == :qtt ? Q.QTTExpression[] : Vector{Float64}[]
        history = Any[]
        accepted = false
        reason = :max_iterations
        check = nothing
        first_energy_iteration = nothing
        for iteration = 1:(scf.max_iterations)
            rec.scf = iteration
            rec.inner = 0
            start = time_ns()
            used_precision_factor = precision_factor
            accuracy = iteration_accuracy(problem, base_accuracy, scf, precision_factor)
            cfg = solver_config(problem, accuracy, scf)
            cfg = merge(cfg, (; mixing = mixing_parameter))
            # Mix the current orbital density, then hold H fixed during the inner orbital updates.
            # The previous output density already represents these unchanged incoming orbitals.
            orbital_density =
                output_density === nothing ?
                density(problem, state.orbitals, accuracy, rec; renormalize = true) : output_density
            rho, mixing = Q.mix_density(
                previous_density,
                orbital_density,
                density_history,
                mixing_residuals,
                cfg,
                rec;
                preconditioner,
            )
            # Hartree is the first consumer requiring independent density cores.
            rho = Q.reconstruct_density(rho, cfg, rec)
            H = effective_hamiltonian(
                problem,
                rho,
                ops,
                accuracy,
                rec;
                fit_potential = solver != :block_lanczos,
            )
            # Only the fixed potential survives all inner stages. Orbitals get
            # new evaluators after every update; a new SCF H gets a new cache.
            potential = Q.timed(rec, "fixed_potential_setup") do
                solver in (:lrrap, :block_lanczos) ? Q.potential_samples(H, rec) :
                Q.memoized_sample(H.potential, rec)
            end
            # Share one synchronized field across the subspace's sampled actions.
            subspace_field = solver == :lrrap ? nothing : Q.synchronized(potential)
            subspace = nothing
            metrics = nothing
            for inner_iteration = 1:(scf.inner_steps)
                rec.inner = inner_iteration
                if solver == :lrrap
                    orbitals, search_directions, metrics = Q.lrrap_step(
                        orbitals,
                        search_directions,
                        H,
                        green,
                        cfg,
                        rec;
                        work_tol = accuracy.work_tolerance,
                        potential,
                    )
                elseif solver == :block_lanczos
                    orbitals, subspace, metrics = Q.block_lanczos_step(
                        orbitals,
                        subspace,
                        H,
                        cfg,
                        rec;
                        work_tol = min(accuracy.work_tolerance, cfg.eigen_tolerance / 10),
                        ritz_tol = cfg.orthogonalization_tolerance,
                        potential = subspace_field,
                        extract_ritz = inner_iteration == scf.inner_steps,
                    )
                else
                    orbitals, subspace, metrics = Q.block_davidson_step(
                        orbitals,
                        subspace,
                        H,
                        green,
                        cfg,
                        rec;
                        work_tol = min(accuracy.work_tolerance, cfg.eigen_tolerance / 10),
                        ritz_tol = cfg.orthogonalization_tolerance,
                        potential = subspace_field,
                    )
                end
                Q.event!(solver_io, merge((scf = iteration, inner = inner_iteration), metrics))
            end
            if representation == :qtt
                Q.event!(
                    rec.stage_io,
                    (;
                        scf = iteration,
                        inner = 0,
                        stage = "fixed_potential_cache",
                        parent_stage = rec.stage,
                        seconds = 0.0,
                        gc_seconds = 0.0,
                        allocated_bytes = 0,
                        sampling_method = Q.sampling_method(potential),
                        requests = potential.requests,
                        evaluations = potential.evaluations,
                        cache_evictions = potential.cache.evictions,
                        tt_cache_constructed = Q.uses_tt_cache(potential),
                    ),
                )
            end
            # Compress to storage accuracy and recompute the density from the stored orbitals.
            rec.inner = 0
            orbitals, compression = Q.compress_orbitals(
                orbitals,
                cfg;
                tol = accuracy.storage_tolerance,
                input_gram_defect = metrics.gram_defect,
                occupied_count,
            )
            stored_gram_defect = compression.stored_occupied_gram_defect
            working_gram_defect = compression.stored_gram_defect
            state = OrbitalState(problem, orbitals[1:occupied_count])
            output_density = density(
                problem,
                state.orbitals,
                accuracy,
                rec;
                renormalize = true,
                label = "output_density",
            )
            canonical = (final_output_ritz = false,)
            energy = total_energy(problem, state.orbitals, output_density, ops, accuracy, rec)
            density_residual =
                Q.difference_norm(output_density, rho; rec, label = "scf_density_residual") /
                Q.ttnorm(output_density; cfg)
            change = isfinite(previous_energy) ? abs(energy.total - previous_energy) : Inf
            energy_stable = change < scf.energy_tolerance && precision_factor == 1
            if energy_stable && first_energy_iteration === nothing
                first_energy_iteration = iteration
            end
            requested_stop = dir !== nothing && isfile(joinpath(dir, "STOP"))
            final_iteration = iteration == scf.max_iterations || requested_stop
            # Energy stability only schedules the independent final-state check.
            if final_iteration || (energy_stable && density_residual <= scf.density_tolerance)
                final_accuracy = iteration_accuracy(problem, base_accuracy, scf)
                candidate = Q.timed(rec, "scf_candidate_verification") do
                    Q.recorded_diagnostic(rec) do diagnostic_rec
                        final_scf_candidate(
                            state,
                            output_density,
                            rho,
                            ops,
                            final_accuracy,
                            scf,
                            audit,
                            physics,
                            diagnostic_rec,
                        )
                    end
                end
                accepted = accept_scf_state(candidate.audit, scf, audit)
                Q.event!(
                    rec.stage_io,
                    (;
                        scf = iteration,
                        inner = 0,
                        stage = "scf_candidate_acceptance",
                        parent_stage = rec.stage,
                        seconds = 0.0,
                        event = "candidate_acceptance",
                        accepted,
                        max_residual = candidate.audit.max_residual,
                        density_residual = candidate.audit.density_residual,
                        gram_defect = candidate.audit.gram_defect,
                    ),
                )
                if accepted || final_iteration
                    # A rejected trial leaves the full working block and its histories intact.
                    state = candidate.state
                    output_density = candidate.density
                    energy = candidate.energy
                    density_residual = candidate.audit.density_residual
                    change = abs(energy.total - previous_energy)
                    compression = candidate.compression
                    stored_gram_defect = compression.stored_gram_defect
                    check = candidate.audit
                    canonical = candidate.metrics
                end
            end
            row = merge(
                (
                    scf = iteration,
                    representation,
                    seconds = (time_ns() - start) / 1e9,
                    energy_change = isfinite(change) ? change : nothing,
                    has_previous_energy = isfinite(previous_energy),
                    density_residual = density_residual,
                    mixing_method = mixing,
                    mixing_parameter = mixing_parameter,
                    precision_factor = used_precision_factor,
                    orbital_fit_tolerance = accuracy.orbital_tolerance,
                    field_fit_tolerance = accuracy.field_tolerance,
                    kinetic_action_tolerance = accuracy.kinetic_action_tolerance,
                    density_fit_tolerance = accuracy.density_tolerance,
                    orbital_rank_cap = accuracy.orbital_cap,
                    occupied_orbitals = occupied_count,
                    extra_orbitals = scf.extra_orbitals,
                    computed_orbitals = orbital_count,
                    first_energy_iteration,
                ),
                energy,
                metrics,
                compression,
                canonical,
                (
                    gram_defect = stored_gram_defect,
                    working_gram_defect,
                    max_rank = Q.orbital_max_rank(state.orbitals),
                    orbital_bytes = sum(Q.storage, state.orbitals),
                    orbital_bond_profiles = representation == :qtt ?
                                            [
                        [size(core, 3) for core in psi.c[1:(end - 1)]] for psi in state.orbitals
                    ] : nothing,
                    bond_profile_order = representation == :qtt ? "LSB to MSB" : nothing,
                ),
            )
            push!(history, row)
            Q.event!(sio, row)
            dir === nothing || save_checkpoint(
                joinpath(dir, "checkpoint.jls"),
                state;
                metadata = Dict(
                    "history" => history,
                    "restart" => "warm_start",
                    "pivot_state_file" => "cross-state.jls",
                    "input_density" => representation == :qtt ? rho.c : rho,
                ),
            )
            dir === nothing ||
                representation == :dense ||
                accepted ||
                final_iteration ||
                Q.save_cross_state(rec, joinpath(dir, "cross-state.jls"))
            @printf(
                "SCF %3d E=% .10f dE=%.3e rank=%s\n",
                iteration,
                energy.total,
                change,
                row.max_rank === nothing ? "n/a" : string(row.max_rank)
            )
            flush(stdout)
            previous_density = rho
            previous_energy = energy.total
            if accepted || final_iteration
                reason = accepted ? :accepted : requested_stop ? :requested_stop : :max_iterations
                break
            end
            if base_accuracy.adaptive
                precision_factor =
                    min(precision_factor, max(1.0, density_residual / (10scf.density_tolerance)))
                change < 10scf.energy_tolerance && (precision_factor = 1.0)
                if precision_factor < used_precision_factor
                    previous_energy = Inf
                    output_density = nothing
                    empty!(density_history)
                    empty!(mixing_residuals)
                    best_density_residual = Inf
                end
            end
            if scf.adaptive_mixing &&
               precision_factor == used_precision_factor &&
               isfinite(previous_density_residual)
                old_mixing = mixing_parameter
                # Detect accumulated growth as well as a single large increase.
                reference_residual = min(best_density_residual, previous_density_residual)
                restart = density_residual > 1.5reference_residual
                if restart
                    mixing_parameter = max(scf.mixing / 16, mixing_parameter / 2)
                    empty!(density_history)
                    empty!(mixing_residuals)
                    best_density_residual = density_residual
                elseif density_residual < 0.5previous_density_residual
                    mixing_parameter = min(scf.mixing, 1.2mixing_parameter)
                    best_density_residual = density_residual
                end
                if restart || mixing_parameter != old_mixing
                    Q.event!(
                        rec.stage_io,
                        (;
                            scf = iteration,
                            inner = 0,
                            stage = "density_mixing_response",
                            event = "mixing_response",
                            seconds = 0.0,
                            previous_residual = previous_density_residual,
                            reference_residual,
                            density_residual,
                            previous_mixing = old_mixing,
                            next_mixing = mixing_parameter,
                            anderson_restarted = restart,
                        ),
                    )
                end
            end
            precision_changed = precision_factor < used_precision_factor
            previous_density_residual = precision_changed ? Inf : density_residual
            best_density_residual =
                precision_changed ? Inf : min(best_density_residual, density_residual)
        end
        elapsed = (time_ns() - t0) / 1e9
        dir === nothing || write_json(
            joinpath(dir, "status.json"),
            (
                status = "finished",
                accepted,
                solver = string(solver),
                representation = string(representation),
                stop_reason = string(reason),
                iterations = length(history),
                first_energy_iteration,
                elapsed_seconds = elapsed,
                final = last(history),
                audit = check,
            ),
        )
        SCFResult(state, history, accepted, reason, check, elapsed)
    catch e
        fail_run(dir, e)
        rethrow()
    finally
        close_run_io(sio)
        close_run_io(solver_io)
        close(rec)
    end
end

coarse_grid(g::PeriodicGrid1D, h) = PeriodicGrid1D(min(g.R, max(2, ceil(Int, log2(g.L / h)))), g.L)
coarse_grid(g::BoxGrid3D, h) =
    BoxGrid3D(ntuple(a -> min(axis_bits(g)[a], max(2, ceil(Int, log2(g.L[a] / h)))), 3), g.L)
next_grid(g::PeriodicGrid1D, target, bits) = PeriodicGrid1D(min(target.R, g.R + bits), target.L)
next_grid(g::BoxGrid3D, target, bits) =
    BoxGrid3D(min.(axis_bits(target), axis_bits(g) .+ bits), target.L)

"Solve density feedback on increasing grids; only the target grid supplies the final result."
function ground_state_refined(
    problem;
    representation = :qtt,
    accuracy,
    scf,
    solver,
    tci,
    guess,
    physics,
    audit,
    refinement,
    output,
)
    stage_scf = SCFOptions(; merge(option_tuple(scf), (; initial_spacing = nothing))...)
    grid = coarse_grid(problem.grid, scf.initial_spacing)
    initial = nothing
    stages = Any[]
    while grid != problem.grid
        stage_problem = Problem(grid, problem.model, problem.occupations; mass = problem.mass)
        coarse_scf = SCFOptions(;
            merge(
                option_tuple(stage_scf),
                (;
                    energy_tolerance = max(scf.energy_tolerance, scf.coarse_energy_tolerance),
                    density_tolerance = max(scf.density_tolerance, scf.coarse_density_tolerance),
                    orbital_residual_tolerance = max(
                        something(scf.orbital_residual_tolerance, scf.eigen_tolerance),
                        scf.coarse_density_tolerance,
                    ),
                ),
            )...,
        )
        stage_output = output === nothing ? nothing : output * "-grid-$(nsites(grid))"
        result = ground_state(
            stage_problem;
            representation,
            initial,
            accuracy,
            scf = coarse_scf,
            solver,
            tci,
            guess,
            physics,
            audit,
            output = stage_output,
        )
        push!(
            stages,
            (;
                qtt_sites = nsites(grid),
                grid_spacing = spacing(grid),
                accepted = result.accepted,
                output = stage_output,
                iterations = length(result.history),
            ),
        )
        result.accepted || error("Coarse-grid SCF was not accepted; retained output: $stage_output")
        grid = next_grid(grid, problem.grid, min(scf.refinement_bits, refinement.max_added_bits))
        next_problem = Problem(grid, problem.model, problem.occupations; mass = problem.mass)
        next_scf = grid == problem.grid ? stage_scf : coarse_scf
        refined_accuracy = iteration_accuracy(next_problem, accuracy, next_scf)
        initial = refine_state(
            result.state,
            next_problem;
            accuracy = refined_accuracy,
            tci,
            refinement,
            parallel_tasks = scf.parallel_tasks,
        )
    end
    result = ground_state(
        problem;
        representation,
        initial,
        accuracy,
        scf = stage_scf,
        solver,
        tci,
        guess,
        physics,
        audit,
        output,
    )
    output === nothing || write_json(
        joinpath(output, "grid-continuation.json"),
        (; stages, target_qtt_sites = nsites(problem.grid)),
    )
    result
end

"Canonicalize occupied orbitals using their output-density Hamiltonian."
function canonicalize_output_orbitals(problem, orbitals, output_density, ops, accuracy, cfg, rec)
    output_hamiltonian = effective_hamiltonian(problem, output_density, ops, accuracy, rec)
    # Use the complete output-density Hamiltonian for the final Ritz matrix.
    canonical_orbitals, offdiag, gram_defect =
        Q.ritz_rotate(orbitals, output_hamiltonian, cfg, rec; work_tol = 0.0, force = true)
    canonical_orbitals, compression = Q.compress_orbitals(
        canonical_orbitals,
        cfg;
        tol = accuracy.storage_tolerance,
        input_gram_defect = gram_defect,
    )
    rho = density(
        problem,
        canonical_orbitals,
        accuracy,
        rec;
        renormalize = true,
        label = "canonical_output_density",
    )
    density_change =
        Q.difference_norm(rho, output_density; rec, label = "ritz_density_change") /
        Q.ttnorm(output_density)
    canonical_orbitals,
    rho,
    compression,
    (
        final_output_ritz = true,
        final_ritz_offdiagonal_before = offdiag,
        final_ritz_density_relative_change = density_change,
    )
end

"Form and measure the exact orbital state that will be saved if accepted or stopped."
function final_scf_candidate(
    state,
    output_density,
    input_density,
    ops,
    accuracy,
    scf,
    audit,
    physics,
    rec,
)
    problem = state.problem
    cfg = solver_config(problem, accuracy, scf)
    orbitals, rho, compression, metrics = canonicalize_output_orbitals(
        problem,
        state.orbitals,
        output_density,
        ops,
        accuracy,
        cfg,
        rec,
    )
    final_state = OrbitalState(problem, orbitals)
    energy = total_energy(problem, orbitals, rho, ops, accuracy, rec)
    check = audit_state(
        final_state,
        input_density,
        accuracy,
        audit,
        physics,
        rec;
        reference_energy = energy.total,
        scf,
    )
    (; state = final_state, density = rho, energy, compression, metrics, audit = check)
end

"One acceptance decision for the canonical stored state; energy changes are progress only."
function accept_scf_state(check, scf, audit)
    all(isfinite, check.orbital_residuals) &&
        check.max_residual + check.kinetic_action_error_budget <
        something(scf.orbital_residual_tolerance, scf.eigen_tolerance) &&
        check.density_residual <= scf.density_tolerance &&
        check.gram_defect < audit.gram_tolerance &&
        check.charge_error < audit.charge_tolerance &&
        check.energy_difference_from_iteration < scf.energy_tolerance
end

# Recompute H, residuals, Gram, charge, and density errors with the independent audit settings.

function audit_accuracy(base::Accuracy, opts::AuditOptions)
    validate(opts)
    Accuracy(;
        merge(
            option_tuple(base),
            (; (name => opts.rank_multiplier * cap for (name, cap) in pairs(field_caps(base)))...),
            (
                adaptive = false,
                orbital_cap = opts.rank_multiplier * base.orbital_cap,
                work_cap = opts.rank_multiplier * base.work_cap,
                field_cap = opts.rank_multiplier * base.field_cap,
                field_tolerance = min(base.field_tolerance, opts.field_tolerance),
                density_tolerance = min(base.density_tolerance, opts.field_tolerance / 100),
                xc_energy_tolerance = min(base.xc_energy_tolerance, opts.field_tolerance / 100),
                kernel_tolerance = opts.kernel_tolerance,
                external_tolerance = opts.kernel_tolerance,
                algebra_tolerance = min(base.algebra_tolerance, opts.algebra_tolerance),
            ),
        )...,
    )
end

function audit_state(
    state,
    input_density,
    base,
    opts,
    physics,
    rec;
    reference_energy = NaN,
    scf = SCFOptions(),
)
    accuracy = audit_accuracy(base, opts)
    problem = state.problem
    orbitals = state.orbitals
    ops = build_operators(problem, accuracy, rec, physics, Val(representation(state)))
    rho = density(problem, orbitals, accuracy, rec; renormalize = true, label = "audit_density")
    H = effective_hamiltonian(problem, rho, ops, accuracy, rec)
    cfg = solver_config(problem, accuracy, scf)
    potential = Q.synchronized(Q.potential_samples(H, rec))
    overlap = Q.gram(orbitals; cfg)
    norms_squared = real.(diag(overlap))
    measured = Q.recorded_map(Tuple{Float64,Float64}, length(orbitals), cfg, rec) do i, localrec
        psi = orbitals[i]
        orbital_norm = sqrt(norms_squared[i])
        orbital_energy = Q.h_expectation(H, psi) / norms_squared[i]
        residual =
            Q.ttnorm(
                Q.h_residual(
                    H,
                    psi,
                    orbital_energy,
                    cfg,
                    localrec;
                    tol = opts.residual_fit_tolerance,
                    cap = accuracy.work_cap,
                    label = "audit_orbital_residual_$i",
                    potential,
                ),
            ) / orbital_norm
        threshold = something(scf.orbital_residual_tolerance, scf.eigen_tolerance)
        if 0.9threshold < residual < 1.1threshold
            residual =
                Q.ttnorm(
                    Q.h_residual(
                        H,
                        psi,
                        orbital_energy,
                        cfg,
                        localrec;
                        tol = opts.residual_fit_tolerance / 10,
                        cap = accuracy.work_cap,
                        label = "audit_boundary_residual_$i",
                        potential,
                    ),
                ) / orbital_norm
        end
        orbital_energy, residual
    end
    orbital_energies, residuals = first.(measured), last.(measured)
    energy = total_energy(problem, orbitals, rho, ops, accuracy, rec)
    gram_defect = norm(overlap - I)
    charge = dot(problem.occupations, norms_squared)
    density_residual =
        input_density === nothing ? NaN :
        Q.difference_norm(rho, input_density; rec, label = "audit_density_residual") /
        Q.ttnorm(rho; cfg)
    energy_difference = abs(energy.total - reference_energy)
    action_error =
        representation(state) == :qtt && problem.grid isa PeriodicGrid1D ?
        something(accuracy.kinetic_action_tolerance, 0.0) : 0.0
    (;
        max_residual = maximum(residuals),
        rms_residual = sqrt(sum(abs2, residuals) / length(orbitals)),
        orbital_residuals = residuals,
        residual_fit_tolerance = opts.residual_fit_tolerance,
        kinetic_action_error_budget = action_error,
        orbital_energies,
        gram_defect,
        electron_number = charge,
        charge_error = abs(charge - sum(problem.occupations)),
        density_residual,
        energy,
        energy_difference_from_iteration = energy_difference,
    )
end
