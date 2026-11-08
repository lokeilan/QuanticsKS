# SCF density feedback, output Ritz rotation, and independent physical audit.

struct SCFResult
    state::OrbitalState
    history::Vector{Any}
    converged::Bool
    stop_reason::Symbol
    audit::NamedTuple
    elapsed_seconds::Float64
end

"Low-rank SCF with optional parallel orbital fits and contractions; history starts fresh."
function ground_state(
    problem::Problem;
    initial = nothing,
    pivot_state = nothing,
    accuracy = Accuracy(),
    scf = SCFOptions(),
    tci = TCIOptions(),
    guess = InitialGuess(),
    physics = PhysicsNumerics(),
    audit = AuditOptions(),
    output = nothing,
)
    validate(accuracy)
    validate(scf)
    validate(tci)
    validate(physics)
    validate(guess)
    all(==(first(problem.occupations)), problem.occupations) ||
        throw(ArgumentError("The current LRRAP solver requires equal occupations"))
    initial !== nothing &&
        !compatible_problem(initial.problem, problem) &&
        throw(
            ArgumentError("Initial state mismatch; call refine_state explicitly for a finer grid"),
        )
    t0 = time_ns()
    Random.seed!(scf.seed)
    dir = prepare_run(output, "scf", problem; accuracy = accuracy, scf, tci, guess, physics, audit)
    rec = recorder(dir, tci; parallel_tasks = scf.parallel_tasks)
    sio = run_io(dir, "scf.jsonl")
    lio = run_io(dir, "lrrap.jsonl")
    try
        pivot_state === nothing || Q.load_cross_state!(rec, pivot_state)
        cfg = solver_config(problem, accuracy, scf)
        ops = build_operators(problem, accuracy, rec, physics)
        green = green_operator(problem, accuracy, scf, physics, ops)
        state =
            initial === nothing ? initial_state(problem, accuracy, rec, guess) :
            OrbitalState(problem, initial.orbitals)
        dir === nothing || save_checkpoint(
            joinpath(dir, "initial.jls"),
            state;
            metadata = Dict("source" => initial === nothing ? "initial_guess" : "provided_state"),
        )
        orbitals = state.orbitals
        previous_density = nothing
        search_directions = nothing
        previous_energy = Inf
        density_history = Q.QTT[]
        mixing_residuals = Q.QTTExpression[]
        history = Any[]
        converged = false
        reason = :max_iterations
        check = nothing
        for k = 1:(scf.max_iterations)
            rec.scf = k
            rec.inner = 0
            start = time_ns()
            # Mix the current orbital density, then hold H fixed during the inner low-rank updates.
            orbital_density = density(problem, orbitals, accuracy, rec; renormalize = true)
            rho, mixing = Q.mix_density(
                previous_density,
                orbital_density,
                density_history,
                mixing_residuals,
                cfg,
                rec,
            )
            # Hartree is the first consumer requiring independent density cores.
            rho = Q.materialize_density(rho, cfg, rec)
            H = effective_hamiltonian(problem, rho, ops, accuracy, rec)
            # Only the fixed potential survives all inner stages. Orbitals get
            # new evaluators after every update; a new SCF H gets a new cache.
            potential = Q.timed(rec, "fixed_potential_setup") do
                Q.potential_samples(H, rec)
            end
            metrics = nothing
            for j = 1:(scf.inner_steps)
                rec.inner = j
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
                Q.event!(lio, merge((scf = k, inner = j), metrics))
            end
            Q.event!(
                rec.stage_io,
                (;
                    scf = k,
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
                    tt_frames_constructed = Q.uses_tt_frames(potential),
                ),
            )
            # Compress to storage accuracy and recompute the density from the stored orbitals.
            rec.inner = 0
            orbitals, pack = Q.pack_orbitals(
                orbitals,
                cfg;
                tol = accuracy.storage_tolerance,
                input_gram_defect = metrics.gram_defect,
            )
            stored_gram_defect = pack.packed_gram_defect
            state = OrbitalState(problem, orbitals)
            output_density = density(
                problem,
                orbitals,
                accuracy,
                rec;
                renormalize = true,
                label = "output_density",
            )
            canonical = (final_output_ritz = false,)
            # Strict mode rotates with the output H, then clears directions tied to the old orbital basis.
            if scf.criterion == :strict
                orbitals, output_density, canonical =
                    canonicalize_output_orbitals(problem, orbitals, ops, accuracy, cfg, rec)
                state = OrbitalState(problem, orbitals)
                stored_gram_defect = canonical.canonical_gram_defect
                # Search directions belonged to the pre-Ritz orbital basis.
                search_directions = nothing
            end
            energy = total_energy(problem, orbitals, output_density, ops, accuracy, rec)
            density_residual =
                Q.difference_norm(output_density, rho; rec, label = "scf_density_residual") /
                Q.ttnorm(output_density; cfg)
            change = isfinite(previous_energy) ? abs(energy.total - previous_energy) : Inf
            if scf.criterion == :strict
                check = audit_state(
                    state,
                    rho,
                    accuracy,
                    audit,
                    physics,
                    rec;
                    reference_energy = energy.total,
                    scf,
                )
                canonical = merge(
                    canonical,
                    (
                        output_eigen_residual_max_Ha = check.max_residual,
                        output_audit_passed = check.passed,
                    ),
                )
            end
            row = merge(
                (
                    scf = k,
                    seconds = (time_ns() - start) / 1e9,
                    energy_change = isfinite(change) ? change : nothing,
                    has_previous_energy = isfinite(previous_energy),
                    density_residual = density_residual,
                    mixing_method = mixing,
                ),
                energy,
                metrics,
                pack,
                canonical,
                (
                    gram_defect = stored_gram_defect,
                    max_rank = maximum(Q.rank.(orbitals)),
                    stored_max_rank = maximum(Q.rank.(orbitals)),
                    orbital_bytes = sum(Q.storage, orbitals),
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
                ),
            )
            dir === nothing || Q.save_cross_state(rec, joinpath(dir, "cross-state.jls"))
            @printf("SCF %3d E=% .10f dE=%.3e rank=%d\n", k, energy.total, change, row.max_rank)
            flush(stdout)
            previous_density = rho
            previous_energy = energy.total
            # Track energy stopping separately from audit acceptance; strict mode requires both.
            if k >= 2 &&
               change < scf.energy_tolerance &&
               (
                   scf.criterion == :energy || (
                       density_residual < scf.density_tolerance &&
                       check.passed &&
                       row.gram_defect < scf.gram_tolerance
                   )
               )
                if scf.criterion == :energy
                    check = audit_state(
                        state,
                        rho,
                        accuracy,
                        audit,
                        physics,
                        rec;
                        reference_energy = energy.total,
                        scf,
                    )
                end
                if scf.criterion == :energy || check.passed
                    converged = true
                    reason = scf.criterion == :energy ? :energy_tolerance : :strict
                    break
                end
            end
            if dir !== nothing && isfile(joinpath(dir, "STOP"))
                reason = :requested_stop
                break
            end
        end
        # If a strict candidate failed and iterations continued, audit the final state again.
        (check === nothing || (!converged && scf.criterion == :energy)) && (
            check = audit_state(
                state,
                previous_density,
                accuracy,
                audit,
                physics,
                rec;
                reference_energy = last(history).total,
                scf,
            )
        )
        elapsed = (time_ns() - t0) / 1e9
        dir === nothing || write_json(
            joinpath(dir, "status.json"),
            (
                status = "finished",
                converged,
                criterion = string(scf.criterion),
                stop_reason = string(reason),
                iterations = length(history),
                elapsed_seconds = elapsed,
                final = last(history),
                audit = check,
            ),
        )
        SCFResult(state, history, converged, reason, check, elapsed)
    catch e
        fail_run(dir, e)
        rethrow()
    finally
        close_run_io(sio)
        close_run_io(lio)
        close(rec)
    end
end

# Rotate occupied orbitals with H of the output density, then recompute the density change.

"Canonicalize occupied orbitals using their output-density Hamiltonian."
function canonicalize_output_orbitals(problem, orbitals, ops, accuracy, cfg, rec)
    before =
        density(problem, orbitals, accuracy, rec; renormalize = true, label = "ritz_output_density")
    Hout = effective_hamiltonian(problem, before, ops, accuracy, rec)
    # The complete operator path is used for this final projected matrix.
    # No spatial symmetry reduction or dense reference enters the solver.
    ys, offdiag = Q.ritz_rotate(orbitals, Hout, cfg, rec; work_tol = 0.0, force = true)
    ys, pack = Q.pack_orbitals(ys, cfg; tol = accuracy.storage_tolerance)
    after =
        density(problem, ys, accuracy, rec; renormalize = true, label = "canonical_output_density")
    density_change =
        Q.difference_norm(after, before; rec, label = "ritz_density_change") / Q.ttnorm(before)
    ys,
    after,
    (
        final_output_ritz = true,
        final_ritz_offdiagonal_before = offdiag,
        final_ritz_density_relative_change = density_change,
        final_ritz_direction_history_reset = true,
        canonical_gram_defect = pack.packed_gram_defect,
    )
end

# Recompute H, residuals, Gram, charge, and density errors with the independent audit settings.

function audit_accuracy(base::Accuracy, opts::AuditOptions)
    validate(opts)
    Accuracy(;
        merge(
            option_tuple(base),
            (; (name => opts.rank_multiplier * cap for (name, cap) in pairs(field_caps(base)))...),
            (
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
    ops = build_operators(problem, accuracy, rec, physics)
    rho = density(problem, orbitals, accuracy, rec; renormalize = true, label = "audit_density")
    H = effective_hamiltonian(problem, rho, ops, accuracy, rec)
    cfg = solver_config(problem, accuracy, scf)
    ev = Q.synchronized(Q.potential_samples(H, rec))
    measured = Q.recorded_map(Tuple{Float64,Float64}, length(orbitals), cfg, rec) do i, localrec
        x = orbitals[i]
        e = Q.h_expectation(H, x) / real(Q.inner(x, x))
        residual =
            Q.ttnorm(
                Q.h_residual(
                    H,
                    x,
                    e,
                    cfg,
                    localrec;
                    tol = accuracy.algebra_tolerance,
                    cap = accuracy.work_cap,
                    label = "audit_orbital_residual_$i",
                    potential = ev,
                ),
            ) / Q.ttnorm(x)
        e, residual
    end
    eps, residuals = first.(measured), last.(measured)
    energy = total_energy(problem, orbitals, rho, ops, accuracy, rec)
    gd = norm(Q.gram(orbitals; cfg) - I)
    charge = sum(
        problem.occupations .* Q.independent_map(
            i -> real(Q.inner(orbitals[i], orbitals[i])),
            Float64,
            length(orbitals),
            cfg,
        ),
    )
    drho =
        input_density === nothing ? NaN :
        Q.difference_norm(rho, input_density; rec, label = "audit_density_residual") /
        Q.ttnorm(rho; cfg)
    de = abs(energy.total - reference_energy)
    passed =
        all(isfinite, residuals) &&
        maximum(residuals) < scf.eigen_tolerance &&
        gd < opts.gram_tolerance &&
        abs(charge - sum(problem.occupations)) < opts.charge_tolerance &&
        drho < scf.density_tolerance &&
        de < scf.energy_tolerance
    (;
        passed,
        max_residual = maximum(residuals),
        rms_residual = sqrt(sum(abs2, residuals) / length(orbitals)),
        orbital_residuals = residuals,
        orbital_energies = eps,
        gram_defect = gd,
        electron_number = charge,
        density_residual = drho,
        energy,
        energy_difference_from_iteration = de,
    )
end
