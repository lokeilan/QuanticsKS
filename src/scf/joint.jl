# Shared joint/FCC SCF: keep H fixed inside orbital updates and mix density outside.

"Complete joint SCF; a cold run loads the optional dense comparison only after SCF."
function joint_scf(p, c, reference, out; case_index = 1, comparison_reference_loader = nothing)
    ispath(out) && error("fresh result directory required")
    mkpath(out)
    l = p.layout
    tol = c["interpolation"]["tolerances"][case_index]
    cap = c["accuracy"]["orbital_cap"]
    tci = TCIOptions(
        max_iterations = c["interpolation"]["max_iterations"],
        global_pivot_search = c["interpolation"]["global_pivot_search"],
        max_global_pivots = 10,
        initial_random_pivots = 12,
        require_convergence = c["interpolation"]["require_convergence"],
    )
    rec = QuanticsKS.recorder(out, tci)
    Random.seed!(c["interpolation"]["seed"])
    receipt = Dict{String,Any}(
        "complete" => false,
        "accepted" => false,
        "representation" => get(get(c, "representation", Dict()), "kind", "joint_xk"),
        "case_index" => case_index,
        "tolerance" => tol,
        "orbital_cap" => cap,
        "constraint_reconstruction_tolerance" => tol,
        "density_selfconsistency_required" =>
            get(c["validation"], "require_density_selfconsistency", true),
        "stop_policy" => "energy_and_retained_physical_gates",
    )
    history = Any[]
    started = time()
    state = nothing
    write_status() = open(io -> JSON3.write(io, receipt), joinpath(out, "completion.json"), "w")
    write_status()
    try
        cache_slices = get(c["execution"], "cache_slices", 8)
        perturbation = get(c["execution"], "initial_perturbation", 0.002)
        state = initialize_joint(l, c, reference, rec; tol, cap, receipt, progress = write_status)
        serialize(joinpath(out, "initial.jls"), (; state, configuration = c))
        receipt["spatial_shape"] = collect(shape(l))
        receipt["k_shape"] = collect(kshape(l))
        open(io -> JSON3.write(io, c), joinpath(out, "configuration.json"), "w")
        rho = occupied_density(p, state)
        old = Inf
        inputs = Vector{Float64}[]
        errors = Vector{Float64}[]
        energy_stopped = false
        final = nothing
        v = c["validation"]
        o = c["scf"]
        inner = copy(c["inner"])
        inner["states"] == 16 || error(
            "current implementation retains 16 states; residual and previous blocks are temporary trial directions",
        )
        # Update orbitals at fixed potential, then evaluate the density from every k point.
        for step = 1:o["maximum_iterations"]
            rec.scf = step
            fields = physical_fields(p, rho)
            result = frozen_inner(
                p,
                state,
                fields.potential,
                inner,
                rec;
                tol,
                cap,
                cache_slices,
                scf_step = step,
            )
            state = result.state
            final = measure_state(p, state)
            drho = norm(final.density - rho) / norm(final.density)
            change = abs(final.total_energy - old)
            energy_stopped = step >= o["minimum_iterations"] && change <= o["energy_tolerance_Ha"]
            physical = physical_admitted(final, drho, v)
            row = (;
                step,
                electronic_energy = final.electronic_energy,
                total_energy = final.total_energy,
                energy_change = isfinite(change) ? change : nothing,
                density_residual = drho,
                maximum_k_gram = final.maximum_k_gram,
                maximum_k_residual = final.maximum_k_residual,
                charge = final.charge,
                energy_stopped,
                physical,
                inner_energy_stopped = result.energy_stopped,
                storage = storage_report(state),
            )
            push!(history, row)
            open(io -> JSON3.write(io, row), joinpath(out, "scf-$step.json"), "w")
            serialize(
                joinpath(out, "checkpoint.jls"),
                (; state, history, configuration = c, case_index),
            )
            @printf(
                "SCF %d E=%.12f dE=%.3e drho=%.3e residual=%.3e gram=%.3e\n",
                step,
                final.electronic_energy,
                change,
                drho,
                final.maximum_k_residual,
                final.maximum_k_gram
            )
            flush(stdout)
            # Energy stopping also requires the configured physical checks; compare dense results later.
            if energy_stopped && physical
                receipt["accepted"] = true
                break
            end
            rho = density_mix(rho, final.density, inputs, errors; mixing = o["mixing"])
            old = final.total_energy
        end
        receipt["physical_accepted"] = receipt["accepted"]
        comparison_reference = reference
        if comparison_reference === nothing && comparison_reference_loader !== nothing
            println("SCF finished; loading dense reference for final comparison only")
            flush(stdout)
            comparison_reference = comparison_reference_loader()
            receipt["dense_loaded_after_scf"] = true
        end
        comparison =
            comparison_reference === nothing ? nothing :
            compare_dense(p, state, comparison_reference, out)
        receipt["dense_comparison_performed"] = comparison !== nothing
        if comparison !== nothing
            receipt["accepted"] &=
                comparison.density_relative_l2 <= v["dense_density_relative_L2"] &&
                abs(final.electronic_energy - comparison_reference.electronic_energy) <=
                v["dense_energy_error_Ha"] &&
                comparison.maximum_k_subspace <= v["dense_maximum_k_subspace_error"]
            receipt["density_relative_l2"] = comparison.density_relative_l2
            receipt["maximum_k_subspace"] = comparison.maximum_k_subspace
            receipt["dense_energy_error_Ha"] =
                abs(final.electronic_energy - comparison_reference.electronic_energy)
            open(io -> JSON3.write(io, comparison), joinpath(out, "comparison.json"), "w")
        end
        merge!(
            receipt,
            Dict(
                "complete" => true,
                "outer_energy_stopped" => energy_stopped,
                "iterations" => length(history),
                "elapsed_seconds" => time() - started,
                "electronic_energy" => final.electronic_energy,
                "total_energy" => final.total_energy,
                "initial_perturbation" => receipt["initial_perturbation_added_this_run"],
                "maximum_k_residual" => final.maximum_k_residual,
                "maximum_k_gram" => final.maximum_k_gram,
                "charge" => final.charge,
                "peak_rss_bytes" => Sys.maxrss(),
            ),
        )
        serialize(
            joinpath(out, "final.jls"),
            (; state, history, configuration = c, comparison, receipt),
        )
        (; state, history, comparison, receipt)
    catch e
        receipt["error"] = sprint(showerror, e)
        state === nothing || serialize(joinpath(out, "failed-state.jls"), state)
        rethrow()
    finally
        receipt["elapsed_seconds"] = time() - started
        write_status()
        close(rec)
    end
end
