# Dense SCF: build H from the input density, solve orbitals, then check the output density.

function run_cyclic_reference(c::Config, dir::String)
    validate(c)
    Threads.nthreads() == c.julia_threads || error("Worker thread count does not match config")
    BLAS.set_num_threads(c.blas_threads)
    FFTW.set_num_threads(c.fft_threads)
    r = Recorder(dir)
    summary = Dict{String,Any}(
        "status" => "running",
        "accepted_reference" => false,
        "started_utc" => string(now(UTC)),
        "process_id" => getpid(),
        "julia_version" => string(VERSION),
        "machine" => Sys.MACHINE,
        "cpu" => Sys.CPU_NAME,
        "blas" => string(BLAS.get_config()),
        "blas_threads" => BLAS.get_num_threads(),
        "julia_threads" => Threads.nthreads(),
        "fft_threads" => c.fft_threads,
        "config" => config_dict(c),
        "storage_comparison" => memory_plan(c),
        "occupation" => 2,
        "orbital_type" => "ComplexF64",
        "boundary" => "periodic full cell",
        "eigensolver" => "full cyclic FD2 matrix; sparse shift factorization and full-cell Lanczos; no spatial sectors",
        "reference_algorithm" => "FD2 KS SCF, Pulay/DIIS, pure complex member of spectrally degenerate Fermi pair",
        "algorithm_source_chat" => "01a18ddb-bc99-7d81-b9fa-4eb05cc4ebc7",
        "no_spatial_symmetry_reduction" => true,
        "model" => model_description(c),
        "timing_note" => "Cold process. Stage wall includes JIT/GC; these and inverse-action timings are overlapping subsets. RSS is workload-process OS high-water since startup.",
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
        psi, hpsi, work, vh, vxc, rho_out, h, hw =
            stage(r, 0, "allocate_complex_orbitals_fft_arrays") do
                (
                    zeros(ComplexF64, n, no),
                    zeros(ComplexF64, n, no),
                    zeros(ComplexF64, n, no),
                    zeros(n),
                    zeros(n),
                    zeros(n),
                    Hamiltonian(zeros(n), g.dx, Counter()),
                    HartreeWorkspace(n, g.L, c),
                )
            end
        mixer = DensityMixer()
        previous = NaN
        converged = false
        last_it = 0
        values = zeros(k)
        final = Dict{String,Any}()
        pair_ever = false
        open(joinpath(dir, "eigensolver_operations.csv"), "w") do io
            println(
                io,
                "scf,converged_states,lanczos_iterations,inverse_multiplies,linear_solves,linear_solve_seconds,independent_eigen_residual_max_Ha,shift_Ha",
            )
        end
        # Build the potential from the input density; check self-consistency using the output density.
        for it = 1:(c.max_scf)
            last_it = it
            t0 = time_ns()
            stage(r, it, "hartree_input_fft") do
                hartree!(vh, hw, g.rho)
            end
            stage(r, it, "xc_input_evaluation") do
                xc!(vxc, g.rho, g.dx, c, tab)
            end
            stage(r, it, "assemble_input_potential") do
                @. h.potential = g.v + vh + vxc
            end
            E = cyclic_eigenpairs(
                h,
                k,
                r,
                it,
                c.seed,
                c.arpack_tol,
                c.max_eigen;
                shift_margin = c.spectral_shift_margin,
                target_tolerance = c.eigen_tol,
            )
            stage(r, it, "release_eigensolver_temporaries_gc") do
                # The factors and Lanczos workspace have left scope. Collect
                # unreachable temporary storage before the next large cycle.
                # This collection and its finalizers are included in the timing.
                GC.gc()
            end
            copyto!(values, E.values)
            gap = values[no + 1] - values[no]
            pair_used = gap <= c.fermi_pair_tolerance
            pair_ever |= pair_used
            input_values = copy(values[1:no])
            input_res = stage(r, it, "construct_complex_occupied_state_and_check") do
                @views @. psi = complex(E.vectors[:, 1:no]) / sqrt(g.dx)
                # At a degenerate Fermi edge, keep one complex pure state normalized with dx.
                if pair_used
                    @views @. psi[:, no] =
                        (E.vectors[:, no] + im * E.vectors[:, no + 1]) / sqrt(2g.dx)
                    input_values[no] = (values[no] + values[no + 1]) / 2
                end
                mul!(hpsi, h, psi)
                @inbounds for b = 1:no, i = 1:n
                    work[i, b] = hpsi[i, b] - input_values[b] * psi[i, b]
                end
                sqrt(g.dx) * maximum(norm(view(work, :, b)) for b = 1:no)
            end
            stage(r, it, "occupied_density") do
                fill!(rho_out, 0.0)
                @inbounds for b = 1:no, i = 1:n
                    rho_out[i] += 2abs2(psi[i, b])
                end
            end
            # Energy and nonlinear residuals below use the new H built from the output density.
            stage(r, it, "hartree_output_fft") do
                hartree!(vh, hw, rho_out)
            end
            exc = stage(r, it, "xc_output_evaluation") do
                xc!(vxc, rho_out, g.dx, c, tab)
            end
            kinetic = stage(r, it, "kinetic_energy") do
                kinetic!(work, psi, g.dx)
                2g.dx * real(dot(psi, work))
            end
            stage(r, it, "assemble_output_potential") do
                @. h.potential = g.v + vh + vxc
            end
            d = stage(r, it, "nonlinear_residual_gram_charge_energy") do
                mul!(hpsi, h, psi)
                rayleigh = g.dx * (psi'hpsi)
                copyto!(work, hpsi)
                mul!(work, psi, rayleigh, -1.0, 1.0)
                residual = sqrt(g.dx) * norm(work)
                residual_max = sqrt(g.dx) * maximum(norm(view(work, :, b)) for b = 1:no)
                gram = norm(g.dx * (psi'psi) - I)
                charge = abs(g.dx * sum(rho_out) - c.natoms)
                density = norm(rho_out - g.rho) / norm(rho_out)
                external = g.dx * dot(rho_out, g.v)
                eh = 0.5g.dx * dot(rho_out, vh)
                energy = kinetic + external + eh + exc.exchange + exc.correlation + g.eii
                (; residual, residual_max, gram, charge, density, energy, external, eh, rayleigh)
            end
            de = isfinite(previous) ? abs(d.energy - previous) : Inf
            eig_ok = max(input_res, E.true_res) <= c.eigen_tol
            cluster = findall(e -> abs(e - values[no]) <= c.degeneracy_tol, values)
            enough = last(cluster) < k
            # Require all energy, density, residual, Gram, charge, and Fermi-cluster checks to pass.
            converged =
                it >= 2 &&
                eig_ok &&
                enough &&
                d.density <= c.density_tol &&
                d.residual <= c.residual_tol &&
                de <= c.energy_tol &&
                d.gram <= c.gram_tol &&
                d.charge <= c.charge_tol
            summary["status"] =
                converged ? "converged" :
                !eig_ok ? "inner_eigensolver_not_converged" :
                !enough ? "insufficient_extra_states" :
                it == c.max_scf ? "scf_iteration_limit" : "running"
            final = Dict{String,Any}(
                "iteration" => it,
                "total_energy_Ha" => d.energy,
                "energy_change_Ha" => de,
                "kinetic_Ha" => kinetic,
                "external_Ha" => d.external,
                "hartree_Ha" => d.eh,
                "exchange_Ha" => exc.exchange,
                "correlation_Ha" => exc.correlation,
                "ion_ion_Ha" => g.eii,
                "density_relative_L2" => d.density,
                "nonlinear_residual_F_Ha" => d.residual,
                "nonlinear_residual_RMS_Ha" => d.residual / sqrt(no),
                "nonlinear_residual_max_column_Ha" => d.residual_max,
                "gram_F" => d.gram,
                "charge_error" => d.charge,
                "input_eigen_residual_max_Ha" => max(input_res, E.true_res),
                "fermi_gap_Ha" => gap,
                "fermi_pair_used" => pair_used,
                "fermi_pair_used_any_iteration" => pair_ever,
                "fermi_cluster_indices" => cluster,
                "extra_states_cover_fermi_cluster" => enough,
                "input_eigenvalues_Ha" => copy(values),
                "output_occupied_rayleigh_Ha" => real.(diag(d.rayleigh)),
            )
            # Include both matrix sparse buffers and complex orbital arrays. Native
            # factorization and transient Lanczos workspaces are represented by RSS.
            summary["persistent_array_payload_bytes"] = stage(r, it, "array_inventory") do
                inventory(
                    joinpath(dir, "arrays.csv"),
                    (
                        geometry = g,
                        orbitals = psi,
                        hpsi = hpsi,
                        work = work,
                        vh = vh,
                        vxc = vxc,
                        rho_out = rho_out,
                        hamiltonian = h,
                        hartree = hw,
                        real_eigenvectors = E.vectors,
                        cyclic_matrix = E.matrix,
                        eigenvalues = values,
                        xc_table = tab,
                        history_inputs = Tuple(mixer.inputs),
                        history_residuals = Tuple(mixer.residuals),
                    ),
                )
            end
            stage(r, it, "density_" * c.mixing_method * "_mixing") do
                if !converged && eig_ok && enough && it < c.max_scf
                    mix_density!(g.rho, rho_out, c, mixer)
                end
            end
            wall = (time_ns() - t0) * 1e-9
            println(
                r.scf,
                join(
                    (
                        it,
                        wall,
                        E.niter,
                        d.energy,
                        de,
                        d.density,
                        d.residual,
                        d.residual / sqrt(no),
                        d.residual_max,
                        d.gram,
                        d.charge,
                        max(input_res, E.true_res),
                        gap,
                        values[end] - values[end - 1],
                        converged,
                    ),
                    ',',
                ),
            )
            flush(r.scf)
            @printf(
                "SCF %d  E=%.12f Ha  density=%.3e  residual_F=%.3e  gap=%.3e  complex_pair=%s  %.3f s\n",
                it,
                d.energy,
                d.density,
                d.residual,
                gap,
                string(pair_used),
                wall
            )
            flush(stdout)
            summary["final"] = final
            summary["process_peak_rss_bytes"] = Int(Sys.maxrss())
            write_toml(joinpath(dir, "summary.toml"), summary)
            (converged || !eig_ok || !enough) && break
            previous = d.energy
        end
        summary["accepted_reference"] = converged
        summary["completed_scf_iterations"] = last_it
        stage(r, last_it, "current_diagnostic_and_state_write") do
            total_current = 0.0
            @inbounds for b = 1:no, i = 1:n
                total_current +=
                    imag(conj(psi[i, b]) * (psi[mod1(i + 1, n), b] - psi[mod1(i - 1, n), b]))
            end
            summary["integrated_number_current_au"] = total_current
            summary["cell_average_number_current_au"] = total_current / g.L
            summary["output_array_payload_bytes"] = inventory(
                joinpath(dir, "output_arrays.csv"),
                (
                    physical_orbitals = psi,
                    density = rho_out,
                    input_density = g.rho,
                    points = g.x,
                    eigenvalues = values,
                ),
            )
            serialize(
                joinpath(dir, "state.jls"),
                (
                    schema_version = 2,
                    accepted_reference = converged,
                    config = config_dict(c),
                    points = g.x,
                    dx = g.dx,
                    orbitals = psi,
                    density = rho_out,
                    input_density = g.rho,
                    input_eigenvalues = copy(values),
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
