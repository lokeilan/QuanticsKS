# Dense block SCF stores Euclidean-normalized columns in U; output uses physical samples.

function run_reference(c::Config, dir::String)
    c.eigensolver == "cyclic" && return run_cyclic_reference(c, dir)
    validate(c)
    Threads.nthreads() == c.julia_threads || error("Worker thread count does not match config")
    BLAS.set_num_threads(c.blas_threads)
    FFTW.set_num_threads(c.fft_threads)
    r = Recorder(dir)
    summary = Dict{String,Any}(
        "status" => "running",
        "accepted_reference" => false,
        "started_utc" => string(now(UTC)),
        "julia_version" => string(VERSION),
        "blas" => string(BLAS.get_config()),
        "blas_threads" => BLAS.get_num_threads(),
        "julia_threads" => Threads.nthreads(),
        "fft_threads" => c.fft_threads,
        "cpu" => Sys.CPU_NAME,
        "word_size" => Sys.WORD_SIZE,
        "machine" => Sys.MACHINE,
        "config" => config_dict(c),
        "storage_comparison" => memory_plan(c),
        "softening" => 1.0,
        "occupation" => 2,
        "boundary" => "periodic full cell",
        "kinetic" => "second order finite difference with both corner couplings",
        "xc" => "soft Coulomb restricted LSDA; direct exchange quadrature; Helbig correlation",
        "no_spatial_symmetry_reduction" => true,
        "eigensolver" => "full-grid locally optimal block [X,W,P] Rayleigh-Ritz with pivoted-QR search-direction rank checks",
        "timing_note" => "Cold process; stage wall includes GC/JIT; GC and compilation are subsets. Operator/preconditioner counters are subsets of eigensolve. RSS is OS process high-water since process start.",
    )
    write_toml(joinpath(dir, "summary.toml"), summary)
    try
        write_toml(joinpath(dir, "model.toml"), model_description(c))
        tab = stage(r, 0, "xc_model_initialization") do
            prepare_xc(c)
        end
        g = stage(r, 0, "geometry_external_initial_density") do
            geometry(c)
        end
        n, no, k = g.n, c.natoms ÷ 2, c.natoms ÷ 2 + c.extra_states
        h, p, hart, U, work, rho_out, vh, vxc = stage(r, 0, "allocate_arrays_fft_plans") do
            h = Hamiltonian(zeros(n), g.dx, Counter())
            p = KineticPreconditioner(n, g.dx, c.preconditioner_shift)
            hart = HartreeWorkspace(n, g.L, c)
            U = randn(MersenneTwister(c.seed), n, k)
            (h, p, hart, U, zeros(n, k), zeros(n), zeros(n), zeros(n))
        end
        stage(r, 0, "precondition_initial_random_block") do
            ldiv!(work, p, U)
            copyto!(U, work)
        end
        iterator = stage(r, 0, "allocate_block_solver_workspace") do
            BlockWorkspace(U)
        end
        eigvals = zeros(k)
        summary["persistent_array_payload_bytes"] = stage(r, 0, "array_inventory") do
            inventory(
                joinpath(dir, "arrays.csv"),
                (
                    geometry = g,
                    hamiltonian = h,
                    preconditioner = p,
                    hartree = hart,
                    orbitals_U = U,
                    diagnostic_work = work,
                    rho_out = rho_out,
                    vh = vh,
                    vxc = vxc,
                    eigenvalues = eigvals,
                    block_solver = iterator,
                ),
            )
        end
        previous_energy = NaN
        converged = false
        last_iteration = 0
        mixer = DensityMixer()
        final = Dict{String,Any}()
        open(joinpath(dir, "eigensolver_operations.csv"), "w") do io
            println(io, "scf,operation,calls,columns,wall_seconds,scope")
        end
        # Use the input density for H and the previous orbitals as the initial guess.
        for it = 1:(c.max_scf)
            last_iteration = it
            it0 = time_ns()
            stage(r, it, "hartree_input_fft") do
                hartree!(vh, hart, g.rho)
            end
            stage(r, it, "xc_input_evaluation") do
                xc!(vxc, g.rho, g.dx, c, tab)
            end
            stage(r, it, "assemble_input_potential") do
                @. h.potential = g.v + vh + vxc
            end
            reset!(h.counter)
            reset!(p.counter)
            eig = stage(r, it, "block_eigensolve") do
                open(joinpath(dir, @sprintf("eigen_trace_%04d.csv", it)), "w") do io
                    block_eigensolve!(
                        iterator,
                        h,
                        p;
                        tol = c.eigen_tol,
                        maxiter = c.max_eigen,
                        trace_io = io,
                    )
                end
            end
            # U is iterator.X, reused across SCF cycles. No symmetry projection.
            copyto!(eigvals, eig.λ)
            open(joinpath(dir, "eigensolver_operations.csv"), "a") do io
                for (name, counter) in
                    (("H_times_block", h.counter), ("kinetic_preconditioner", p.counter))
                    println(
                        io,
                        join(
                            (
                                it,
                                name,
                                counter.calls,
                                counter.columns,
                                counter.seconds,
                                "subset_of_block_eigensolve",
                            ),
                            ',',
                        ),
                    )
                end
                println(
                    io,
                    join(
                        (
                            it,
                            "solver_initialization",
                            1,
                            k,
                            eig.initialization_seconds,
                            "subset_of_block_eigensolve",
                        ),
                        ',',
                    ),
                )
            end
            input_res = stage(r, it, "independent_input_eigen_residual") do
                mul!(work, h, U)
                for b = 1:k, i = 1:n
                    work[i, b] -= U[i, b] * eigvals[b]
                end
                maximum(norm(view(work, :, b)) for b = 1:k)
            end
            stage(r, it, "occupied_density") do
                density!(rho_out, U, no, g.dx)
            end
            # Rebuild H from the output density to check the nonlinear, rather than input-H, residual.
            stage(r, it, "hartree_output_fft") do
                hartree!(vh, hart, rho_out)
            end
            exc = stage(r, it, "xc_output_evaluation") do
                xc!(vxc, rho_out, g.dx, c, tab)
            end
            kinetic = stage(r, it, "kinetic_energy") do
                kinetic!(view(work, :, 1:no), view(U, :, 1:no), g.dx)
                2dot(view(U, :, 1:no), view(work, :, 1:no))
            end
            stage(r, it, "assemble_output_potential") do
                @. h.potential = g.v + vh + vxc
            end
            diagnostics = stage(r, it, "nonlinear_residual_gram_charge_energy") do
                Q = view(U, :, 1:no)
                W = view(work, :, 1:no)
                mul!(W, h, Q)
                rayleigh = Q' * W
                # Work := H[rho_out] U - U (U' H[rho_out] U).
                mul!(W, Q, rayleigh, -1.0, 1.0)
                residual = norm(W)
                residual_max = maximum(norm(view(W, :, b)) for b = 1:no)
                gram = norm(Q'Q - I)
                charge = abs(g.dx * sum(rho_out) - c.natoms)
                drho = norm(rho_out - g.rho) / norm(rho_out)
                external = g.dx * dot(rho_out, g.v)
                eh = 0.5g.dx * dot(rho_out, vh)
                energy = kinetic + external + eh + exc.exchange + exc.correlation + g.eii
                (; residual, residual_max, gram, charge, drho, energy, external, eh, rayleigh)
            end
            d = diagnostics
            de = isfinite(previous_energy) ? abs(d.energy - previous_energy) : Inf
            gap = eigvals[no + 1] - eigvals[no]
            fermi_cluster = findall(e -> abs(e - eigvals[no]) <= c.degeneracy_tol, eigvals)
            enough_extra_states = last(fermi_cluster) < k
            all(isfinite, (d.energy, d.residual, d.gram, d.charge, d.drho, input_res)) ||
                error("Nonfinite diagnostic")
            eig_ok = eig.converged && input_res <= c.eigen_tol
            converged =
                it >= 2 &&
                eig_ok &&
                enough_extra_states &&
                d.drho <= c.density_tol &&
                d.residual <= c.residual_tol &&
                d.gram <= c.gram_tol &&
                d.charge <= c.charge_tol &&
                de <= c.energy_tol
            status =
                converged ? "converged" :
                !eig_ok ? "inner_eigensolver_not_converged" :
                !enough_extra_states ? "insufficient_extra_states" :
                it == c.max_scf ? "scf_iteration_limit" : "running"
            final = Dict{String,Any}(
                "iteration" => it,
                "total_energy_Ha" => d.energy,
                "kinetic_Ha" => kinetic,
                "external_Ha" => d.external,
                "hartree_Ha" => d.eh,
                "exchange_Ha" => exc.exchange,
                "correlation_Ha" => exc.correlation,
                "ion_ion_Ha" => g.eii,
                "energy_change_Ha" => de,
                "density_relative_L2" => d.drho,
                "nonlinear_residual_F_Ha" => d.residual,
                "nonlinear_residual_RMS_Ha" => d.residual / sqrt(no),
                "nonlinear_residual_max_column_Ha" => d.residual_max,
                "gram_F" => d.gram,
                "charge_error" => d.charge,
                "input_eigen_residual_max_Ha" => input_res,
                "fermi_gap_Ha" => gap,
                "fermi_boundary_near_degenerate" => (gap <= c.degeneracy_tol),
                "fermi_cluster_indices" => fermi_cluster,
                "extra_states_cover_fermi_cluster" => enough_extra_states,
                "extra_block_end_gap_Ha" => (eigvals[end] - eigvals[end - 1]),
                "input_eigenvalues_Ha" => copy(eigvals),
                "output_occupied_rayleigh_Ha" => diag(d.rayleigh),
            )
            stage(r, it, "density_" * c.mixing_method * "_mixing") do
                # Keep the final input density unchanged if returning a state.
                if !converged && eig_ok && enough_extra_states && it < c.max_scf
                    mix_density!(g.rho, rho_out, c, mixer)
                end
            end
            wall = (time_ns() - it0) * 1e-9
            println(
                r.scf,
                join(
                    (
                        it,
                        wall,
                        length(eig.trace),
                        d.energy,
                        de,
                        d.drho,
                        d.residual,
                        d.residual / sqrt(no),
                        d.residual_max,
                        d.gram,
                        d.charge,
                        input_res,
                        gap,
                        eigvals[end] - eigvals[end - 1],
                        converged,
                    ),
                    ',',
                ),
            )
            flush(r.scf)
            @printf(
                "SCF %d  E=%.12f Ha  density=%.3e  residual_F=%.3e  %.3f s\n",
                it,
                d.energy,
                d.drho,
                d.residual,
                wall
            )
            flush(stdout)
            summary["status"] = status
            summary["final"] = final
            summary["process_peak_rss_bytes"] = Int(Sys.maxrss())
            write_toml(joinpath(dir, "summary.toml"), summary)
            (converged || !eig_ok || !enough_extra_states) && break
            previous_energy = d.energy
        end
        summary["accepted_reference"] = converged
        summary["completed_scf_iterations"] = last_iteration
        stage(r, last_iteration, "write_state") do
            summary["density_history_payload_bytes"] = inventory(
                joinpath(dir, "density_history_arrays.csv"),
                (inputs = Tuple(mixer.inputs), residuals = Tuple(mixer.residuals)),
            )
            # Store physical samples psi=U/sqrt(dx) in column order; no gauge rotation.
            psi = Matrix(view(U, :, 1:no))
            psi ./= sqrt(g.dx)
            summary["output_array_payload_bytes"] = inventory(
                joinpath(dir, "output_arrays.csv"),
                (
                    physical_orbitals = psi,
                    density = rho_out,
                    input_density = g.rho,
                    points = g.x,
                    eigenvalues = eigvals,
                    output_rayleigh = final["output_occupied_rayleigh_Ha"],
                ),
            )
            serialize(
                joinpath(dir, "state.jls"),
                (
                    schema_version = 1,
                    accepted_reference = converged,
                    config = config_dict(c),
                    points = g.x,
                    dx = g.dx,
                    orbitals = psi,
                    density = rho_out,
                    input_density = g.rho,
                    input_eigenvalues = copy(eigvals),
                    output_occupied_rayleigh = final["output_occupied_rayleigh_Ha"],
                    diagnostics = final,
                ),
            )
        end
        summary["state_file_bytes"] = filesize(joinpath(dir, "state.jls"))
    catch err
        summary["status"] = "failed"
        summary["accepted_reference"] = false
        summary["error"] = sprint(showerror, err, catch_backtrace())
        rethrow()
    finally
        summary["finished_utc"] = string(now(UTC))
        summary["solver_and_output_wall_seconds"] = (time_ns() - r.origin) * 1e-9
        summary["stage_wall_seconds_sum"] = r.stage_seconds
        summary["process_peak_rss_bytes"] = Int(Sys.maxrss())
        summary["rss_at_end_bytes"] = memory_sample().rss
        write_toml(joinpath(dir, "summary.toml"), summary)
        close_recorder(r)
    end
    return summary
end
