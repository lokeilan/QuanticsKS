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
    min(
        opts.rank_multiplier,
        opts.field_tolerance,
        opts.kernel_tolerance,
        opts.algebra_tolerance,
        opts.gram_tolerance,
        opts.charge_tolerance,
    ) > 0 || throw(ArgumentError("Invalid audit options"))
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
    residuals = Float64[]
    eps = Float64[]
    cfg = solver_config(p, a, scf)
    ev = Q.evaluate(H.potential, rec)
    for x in xs
        e = Q.h_expectation(H, x) / real(Q.inner(x, x))
        push!(eps, e)
        push!(
            residuals,
            Q.ttnorm(
                Q.h_residual(
                    H, x, e, cfg, rec;
                    tol = a.algebra_tolerance,
                    cap = a.work_cap,
                    label = "audit_orbital_residual",
                    potential = ev,
                ),
            ) / Q.ttnorm(x),
        )
    end
    energy = total_energy(p, xs, n, ops, a, rec)
    gd = norm(Q.gram(xs) - I)
    charge = sum(p.occupations .* [real(Q.inner(x, x)) for x in xs])
    drho =
        input_density === nothing ? NaN :
        Q.difference_norm(n, input_density; rec, label = "audit_density_residual") / Q.ttnorm(n)
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
