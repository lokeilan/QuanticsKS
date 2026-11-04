# Recompute H, residuals, Gram, charge, and density errors with the independent audit settings.

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
    validate(opts)
    a = Accuracy(;
        merge(
            option_tuple(base),
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
    p = state.problem
    xs = state.orbitals
    ops = build_operators(p, a, rec, physics)
    n = density(p, xs, a, rec; renormalize = true, label = "audit_density")
    H = effective_hamiltonian(p, n, ops, a, rec)
    cfg = solver_config(p, a, scf)
    ev = Q.synchronized(Q.potential_samples(H, rec))
    measured = Q.recorded_map(Tuple{Float64,Float64}, length(xs), cfg, rec) do i, localrec
        x = xs[i]
        e = Q.h_expectation(H, x) / real(Q.inner(x, x))
        residual =
            Q.ttnorm(
                Q.h_residual(
                    H,
                    x,
                    e,
                    cfg,
                    localrec;
                    tol = a.algebra_tolerance,
                    cap = a.work_cap,
                    label = "audit_orbital_residual_$i",
                    potential = ev,
                ),
            ) / Q.ttnorm(x)
        e, residual
    end
    eps, residuals = first.(measured), last.(measured)
    energy = total_energy(p, xs, n, ops, a, rec)
    gd = norm(Q.gram(xs; cfg) - I)
    charge = sum(
        p.occupations .*
        Q.independent_map(i -> real(Q.inner(xs[i], xs[i])), Float64, length(xs), cfg),
    )
    drho =
        input_density === nothing ? NaN :
        Q.difference_norm(n, input_density; rec, label = "audit_density_residual") /
        Q.ttnorm(n; cfg)
    de = abs(energy.total - reference_energy)
    passed =
        all(isfinite, residuals) &&
        maximum(residuals) < scf.eigen_tolerance &&
        gd < opts.gram_tolerance &&
        abs(charge - sum(p.occupations)) < opts.charge_tolerance &&
        drho < scf.density_tolerance &&
        de < scf.energy_tolerance
    (;
        passed,
        max_residual = maximum(residuals),
        rms_residual = sqrt(sum(abs2, residuals) / length(xs)),
        orbital_residuals = residuals,
        orbital_energies = eps,
        gram_defect = gd,
        electron_number = charge,
        density_residual = drho,
        energy,
        energy_difference_from_iteration = de,
    )
end
