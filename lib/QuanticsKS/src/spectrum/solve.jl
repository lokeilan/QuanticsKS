# Fixed-potential iteration and checkpoints; stop on the configured occupied-energy change.

function spectrum_checkpoint(dir, p, xs, eps, history, o, potential_hash)
    dir === nothing && return
    data = Dict(
        "schema_version" => 1,
        "kind" => "fixed_spectrum",
        "problem" => problem_descriptor(p),
        "cores" => [x.c for x in xs],
        "eigenvalues" => eps,
        "occupied_states" => o.occupied_states,
        "history" => history,
        "potential_sha256" => potential_hash,
    )
    tmp = joinpath(dir, "spectrum.jls.tmp")
    serialize(tmp, data)
    mv(tmp, joinpath(dir, "spectrum.jls"); force = true)
    save_checkpoint(
        joinpath(dir, "occupied-checkpoint.jls"),
        OrbitalState(p, xs[1:(o.occupied_states)]);
        metadata = Dict(
            "fixed_hamiltonian" => true,
            "solved_states" => o.states,
            "selection" => "lowest occupied_states after Rayleigh-Ritz",
        ),
    )
end

"Optimize QTT eigenstates of a supplied fixed potential; only selected states enter the physical density."
function fixed_spectrum(
    initial::OrbitalState,
    potential::AbstractVector{<:Real};
    spectrum = FixedSpectrumOptions(),
    accuracy = Accuracy(),
    solver = SCFOptions(green_polish_every = 1),
    tci = TCIOptions(),
    output = nothing,
    resume_from = nothing,
)
    o = validate(spectrum)
    a = validate(accuracy)
    validate(solver)
    validate(tci)
    p = initial.problem
    g = p.grid
    N = length(g)
    h = spacing(g)
    length(potential) == N && all(isfinite, potential) ||
        throw(ArgumentError("Invalid fixed potential"))
    length(initial.orbitals) == length(p.occupations) == o.occupied_states ||
        throw(ArgumentError("Physical occupations must describe only the selected occupied states"))
    o.states <= N || throw(ArgumentError("More states than grid points"))
    potential_hash = bytes2hex(sha256(reinterpret(UInt8, Float64.(potential))))
    resumed = nothing
    if resume_from !== nothing
        data = deserialize(resume_from)
        get(data, "kind", nothing) == "fixed_spectrum" && get(data, "schema_version", 0) == 1 ||
            throw(ArgumentError("Not a supported spectrum checkpoint"))
        data["problem"] == problem_descriptor(p) &&
            data["occupied_states"] == o.occupied_states &&
            length(data["cores"]) == o.states ||
            throw(ArgumentError("Resume problem or state count mismatch"))
        hash = get(data, "potential_sha256", nothing)
        if hash === nothing
            sidecar = joinpath(dirname(resume_from), "metadata.json")
            isfile(sidecar) ||
                throw(ArgumentError("Legacy spectrum requires its potential-hash metadata sidecar"))
            hash = JSON3.read(read(sidecar, String))["potential_sha256"]
        end
        hash == potential_hash || throw(ArgumentError("Resume fixed potential mismatch"))
        resumed = [Q.QTT(c) for c in data["cores"]]
    end
    t0 = time_ns()
    Random.seed!(o.seed)
    dir = prepare_run(output, "fixed_spectrum", p; spectrum = o, accuracy = a, solver, tci)
    rec = recorder(dir, tci)
    io = run_io(dir, "iterations.jsonl")
    try
        cfg = solver_config(p, a, solver)
        v = Q.timed(rec, "potential_tci") do
            Q.cross(
                rec,
                g.R,
                j -> potential[j + 1];
                tol = a.external_tolerance,
                cap = a.field_cap,
                label = "fixed_potential",
            )
        end
        compression = Q.timed(rec, "potential_diagnostics") do
            ev = Q.evaluate(v, rec)
            err2 = 0.0
            maxerr = 0.0
            for j = 0:(N - 1)
                err = abs(ev(j) - potential[j + 1])
                err2 += err^2
                maxerr = max(maxerr, err)
            end
            (
                max_absolute_error = maxerr,
                rms_absolute_error = sqrt(err2 / N),
                relative_l2 = sqrt(err2 / sum(abs2, potential)),
                max_bond = Q.rank(v),
                bytes = Q.storage(v),
            )
        end
        Q.timed(rec, "potential_output") do
            if dir !== nothing
                write_json(joinpath(dir, "potential-compression.json"), compression)
                serialize(joinpath(dir, "potential-cores.jls"), v.c)
            end
        end
        H, green = Q.timed(rec, "operator_construction") do
            H = Q.Hamiltonian(
                v,
                Q.diagonal(v),
                Q.circulant(Q.delta_tt(g.R, 1)),
                Q.circulant(Q.delta_tt(g.R, N - 1)),
                h * sqrt(p.mass),
            )
            H, Q.green_mpo(g.R, H.h, solver.green_shift)
        end
        xs = if resumed === nothing
            Q.timed(rec, "initial_orbitals") do
                envelope = Q.cross(
                    rec,
                    g.R,
                    j -> exp(
                        o.virtual_envelope_amplitude * cos(2pi * j * h / o.virtual_envelope_period),
                    );
                    tol = o.ritz_tolerance,
                    cap = a.orbital_cap,
                    label = "virtual_envelope",
                )
                vcat(
                    initial.orbitals,
                    [Q.normalize(Q.phase_tt(envelope, k)) for k in o.virtual_wavevectors],
                )
            end
        else
            resumed
        end
        initial_tol =
            o.adaptive_ritz ? max(o.ritz_tolerance, o.max_ritz_tolerance) : o.ritz_tolerance
        initial_cfg = merge(cfg, (ortho_tol = initial_tol,))
        if resumed === nothing
            xs = Q.timed(rec, "initial_orthogonalization") do
                first(Q.orthogonalize(xs, initial_cfg, rec; force = true))
            end
            xs, eps = spectrum_ritz(xs, H, cfg, rec, o, potential; tol = initial_tol)
        else
            eps = zeros(o.states)
        end
        history = Any[]
        previous = nothing
        converged = false
        for iteration = 0:(o.max_iterations)
            step_start = time_ns()
            metrics = nothing
            if iteration > 0
                rec.scf = iteration
                rec.inner = 1
                if o.method == :block_green
                    xs, metrics = simple_green_step(
                        xs,
                        last(history).eigenvalues,
                        H,
                        green,
                        cfg,
                        rec,
                        o,
                        potential,
                        a,
                    )
                elseif o.method == :lrrap
                    xs, previous, metrics =
                        Q.lrrap_step(xs, previous, H, green, cfg, rec; work_tol = a.work_tolerance)
                    # Rotation changes state identities; reset rather than mismatch columns.
                    xs, eps = spectrum_ritz(xs, H, cfg, rec, o, potential)
                    previous = nothing
                else
                    old_xs = xs
                    work =
                        o.adaptive_work ?
                        max(
                            a.work_tolerance,
                            min(
                                o.max_work_tolerance,
                                o.work_residual_factor * last(history).max_residual,
                            ),
                        ) : a.work_tolerance
                    ritz =
                        o.adaptive_ritz ?
                        max(
                            o.ritz_tolerance,
                            min(
                                o.max_ritz_tolerance,
                                o.ritz_residual_factor * H.h^2 * last(history).max_residual,
                            ),
                        ) : o.ritz_tolerance
                    xs, metrics = residual_subspace_step(
                        xs,
                        H,
                        green,
                        cfg,
                        rec,
                        o,
                        potential;
                        work_tolerance = work,
                        ritz_tolerance = ritz,
                        previous,
                        energies = last(history).eigenvalues,
                        diagnostic_residuals = last(history).residuals,
                    )
                    previous = old_xs
                end
            end
            rec.stage = "diagnostics"
            eps, residuals = Q.timed(rec, "residual_diagnostics") do
                spectrum_residuals(xs, H, cfg, o, rec, potential)
            end
            gram_all, gram_occ = Q.timed(rec, "overlap_diagnostics") do
                norm(Q.gram(xs) - I), norm(Q.gram(xs[1:(o.occupied_states)]) - I)
            end
            occupied_band_energy, energy_change = Q.timed(rec, "energy_diagnostics") do
                e = dot(p.occupations, eps[1:(o.occupied_states)])
                e, isempty(history) ? nothing : abs(e - last(history).occupied_band_energy)
            end
            row = (
                iteration,
                seconds = (time_ns() - step_start) / 1e9,
                elapsed_seconds = (time_ns() - t0) / 1e9,
                eigenvalues = eps,
                residuals,
                max_residual = maximum(residuals),
                rms_residual = sqrt(sum(abs2, residuals) / o.states),
                occupied_max_residual = maximum(residuals[1:(o.occupied_states)]),
                occupied_rms_residual = sqrt(
                    sum(abs2, residuals[1:(o.occupied_states)]) / o.occupied_states,
                ),
                gram_defect = gram_all,
                occupied_gram_defect = gram_occ,
                trace = sum(eps),
                occupied_trace = sum(eps[1:(o.occupied_states)]),
                occupied_band_energy,
                energy_change,
                max_bond = maximum(Q.rank.(xs)),
                orbital_bytes = sum(Q.storage, xs),
                update = metrics,
            )
            push!(history, row)
            Q.event!(io, row)
            Q.timed(rec, "checkpoint_output") do
                spectrum_checkpoint(dir, p, xs, eps, history, o, potential_hash)
            end
            @printf(
                "FIXED-H %3d states=%d occupied=%d maxres=%.3e occmax=%.3e rms=%.3e Gram=%.3e trace=%.10f rank=%d time=%.2fs\n",
                iteration,
                o.states,
                o.occupied_states,
                row.max_residual,
                row.occupied_max_residual,
                row.rms_residual,
                gram_all,
                row.trace,
                row.max_bond,
                row.seconds
            )
            flush(stdout)
            energy_change === nothing || (
                @printf(
                    "    occupied fixed-H band energy=%.12f Ha delta=%.3e Ha criterion=energy\n",
                    occupied_band_energy,
                    energy_change
                );
                flush(stdout)
            )
            if spectrum_accepted(o, iteration, energy_change)
                converged = true
                break
            end
            dir !== nothing && isfile(joinpath(dir, "STOP")) && break
        end
        elapsed = (time_ns() - t0) / 1e9
        dir === nothing || write_json(
            joinpath(dir, "status.json"),
            (
                status = "finished",
                converged,
                criterion = "energy",
                iterations = last(history).iteration,
                elapsed_seconds = elapsed,
                potential_compression = compression,
                final = last(history),
                fixed_hamiltonian = true,
                solved_states = o.states,
                occupied_states = o.occupied_states,
                electrons = sum(p.occupations),
                canonical_potential = o.method == :lrrap ?
                                      "compressed potential in LRRAP update; original input in Ritz and diagnostics" :
                                      "original input samples in updates, Rayleigh-Ritz and residual diagnostics",
                method = string(o.method),
            ),
        )
        FixedSpectrumResult(
            p,
            xs,
            eps,
            OrbitalState(p, xs[1:(o.occupied_states)]),
            history,
            converged,
            elapsed,
        )
    catch e
        fail_run(dir, e)
        rethrow()
    finally
        close_run_io(io)
        close(rec)
    end
end
