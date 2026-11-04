# SCF loop; scf_audit.jl checks output states and scf_output_ritz.jl rotates their orbitals.

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
    p::Problem;
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
    a = validate(accuracy)
    validate(scf)
    validate(tci)
    validate(physics)
    validate(guess)
    all(==(first(p.occupations)), p.occupations) ||
        throw(ArgumentError("The current LRRAP solver requires equal occupations"))
    initial !== nothing &&
        !compatible_problem(initial.problem, p) &&
        throw(
            ArgumentError("Initial state mismatch; call refine_state explicitly for a finer grid"),
        )
    t0 = time_ns()
    Random.seed!(scf.seed)
    dir = prepare_run(output, "scf", p; accuracy = a, scf, tci, guess, physics, audit)
    rec = recorder(dir, tci; parallel_tasks = scf.parallel_tasks)
    sio = run_io(dir, "scf.jsonl")
    lio = run_io(dir, "lrrap.jsonl")
    try
        pivot_state === nothing || Q.load_cross_state!(rec, pivot_state)
        cfg = solver_config(p, a, scf)
        ops = build_operators(p, a, rec, physics)
        green = Q.green_mpo(p.grid.R, spacing(p.grid) * sqrt(p.mass), scf.green_shift)
        state =
            initial === nothing ? initial_state(p, a, rec, guess) :
            OrbitalState(p, initial.orbitals)
        dir === nothing || save_checkpoint(
            joinpath(dir, "initial.jls"),
            state;
            metadata = Dict("source" => initial === nothing ? "initial_guess" : "provided_state"),
        )
        xs = state.orbitals
        oldinput = nothing
        previous = nothing
        oldenergy = Inf
        inputs = Q.QTT[]
        residuals = Q.QTTExpression[]
        history = Any[]
        converged = false
        reason = :max_iterations
        check = nothing
        for k = 1:(scf.max_iterations)
            rec.scf = k
            rec.inner = 0
            start = time_ns()
            # Mix the current orbital density, then hold H fixed during the inner low-rank updates.
            nin = density(p, xs, a, rec; renormalize = true)
            n, mixing = Q.mix_density(oldinput, nin, inputs, residuals, cfg, rec)
            # Hartree is the first consumer requiring independent density cores.
            n = Q.materialize_density(n, cfg, rec)
            H = effective_hamiltonian(p, n, ops, a, rec)
            # Only the fixed potential survives all inner stages. Orbitals get
            # new evaluators after every update; a new SCF H gets a new cache.
            potential = Q.timed(rec, "fixed_potential_setup") do
                Q.potential_samples(H, rec)
            end
            metrics = nothing
            for j = 1:(scf.inner_steps)
                rec.inner = j
                xs, previous, metrics = Q.lrrap_step(
                    xs,
                    previous,
                    H,
                    green,
                    cfg,
                    rec;
                    work_tol = a.work_tolerance,
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
            xs, pack = Q.pack_orbitals(
                xs,
                cfg;
                tol = a.storage_tolerance,
                input_gram_defect = metrics.gram_defect,
            )
            stored_gram_defect = pack.packed_gram_defect
            state = OrbitalState(p, xs)
            nout = density(p, xs, a, rec; renormalize = true, label = "output_density")
            canonical = (final_output_ritz = false,)
            # Strict mode rotates with the output H, then clears directions tied to the old orbital basis.
            if scf.criterion == :strict
                xs, nout, canonical = canonicalize_output_orbitals(p, xs, ops, a, cfg, rec)
                state = OrbitalState(p, xs)
                stored_gram_defect = canonical.canonical_gram_defect
                # Search directions belonged to the pre-Ritz orbital basis.
                previous = nothing
            end
            energy = total_energy(p, xs, nout, ops, a, rec)
            drho =
                Q.difference_norm(nout, n; rec, label = "scf_density_residual") /
                Q.ttnorm(nout; cfg)
            change = isfinite(oldenergy) ? abs(energy.total - oldenergy) : Inf
            if scf.criterion == :strict
                check = audit_state(
                    state,
                    n,
                    a,
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
                    has_previous_energy = isfinite(oldenergy),
                    density_residual = drho,
                    mixing_method = mixing,
                ),
                energy,
                metrics,
                pack,
                canonical,
                (
                    gram_defect = stored_gram_defect,
                    max_rank = maximum(Q.rank.(xs)),
                    orbital_bytes = sum(Q.storage, xs),
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
            oldinput = n
            oldenergy = energy.total
            # Track energy stopping separately from audit acceptance; strict mode requires both.
            if k >= 2 &&
               change < scf.energy_tolerance &&
               (
                   scf.criterion == :energy || (
                       drho < scf.density_tolerance &&
                       check.passed &&
                       row.gram_defect < scf.gram_tolerance
                   )
               )
                if scf.criterion == :energy
                    check = audit_state(
                        state,
                        n,
                        a,
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
                oldinput,
                a,
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
