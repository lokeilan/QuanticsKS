# Green/residual subspace updates at fixed H, with diagnostics for the updated orbitals.

"One Green update per orbital followed by one Rayleigh–Ritz in the updated block."
function simple_green_step(xs, energies, H, green, cfg, rec, o, potential, a)
    ys = Q.QTT[]
    rec.stage = "block_green"
    for (i, x) in enumerate(xs)
        rhs = Q.timed(rec, "green_rhs"; item = string(i)) do
            ex = Q.evaluate(x, rec)
            Q.cross(
                rec,
                cfg.R,
                j -> (cfg.green_shift + energies[i] - potential[j + 1]) * ex(j);
                tol = a.green_rhs_tolerance,
                cap = a.work_cap,
                label = "simple_green_rhs_$i",
            )
        end
        y = Q.timed(rec, "green_mpo"; item = string(i)) do
            Q.apply_mpo(green, rhs; tol = a.algebra_tolerance, cap = a.orbital_cap)
        end
        y = Q.timed(rec, "orbital_normalization"; item = string(i)) do
            Q.ttnorm(y) > 1e-14 || error("Green update produced a zero orbital")
            Q.normalize(y)
        end
        push!(ys, y)
    end
    rec.stage = "rayleigh_ritz"
    ys, eps = spectrum_ritz(ys, H, cfg, rec, o, potential; tol = o.ritz_tolerance)
    ys,
    (
        trial_dimension = length(xs),
        rhs_tolerance = a.green_rhs_tolerance,
        green_tolerance = a.algebra_tolerance,
        ritz_tolerance = o.ritz_tolerance,
        search_basis = "updated orbitals only",
    )
end

function spectrum_accepted(o, iteration, energy_change)
    iteration >= o.min_iterations &&
        energy_change !== nothing &&
        isfinite(energy_change) &&
        0 <= energy_change < o.energy_tolerance
end

"Green-preconditioned residual expansion; all trial and accepted vectors remain QTT."
function residual_subspace_step(
    xs,
    H,
    green,
    cfg,
    rec,
    o,
    potential;
    work_tolerance,
    ritz_tolerance = o.ritz_tolerance,
    previous = nothing,
    energies = nothing,
    diagnostic_residuals = nothing,
)
    rec.stage = "residual_actions"
    projected_H = spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = work_tolerance)
    zs = Q.QTT[]
    for (i, x) in enumerate(xs)
        e = Q.h_expectation(projected_H, x) / real(Q.inner(x, x))
        r = Q.h_residual(
            projected_H, x, e, cfg, rec;
            tol = work_tolerance,
            cap = cfg.workcap,
            label = "fixed_search_residual_$i",
            potential = j -> potential[j + 1],
        )
        # Search-direction filtering only; this is not a convergence criterion.
        Q.ttnorm(r) > o.direction_residual_floor || continue
        # Green removes high-frequency action error; its own truncation must not
        # reintroduce it at the much looser action interpolation tolerance.
        green_tol =
            o.search_basis == :direct ?
            min(work_tolerance, o.green_tolerance_factor * ritz_tolerance) : work_tolerance
        z = Q.apply_mpo(green, r; tol = green_tol, cap = cfg.workcap)
        Q.ttnorm(z) > 1e-14 && push!(zs, Q.normalize(z))
    end
    residual_count = length(zs)
    o.use_previous_subspace && previous !== nothing && append!(zs, previous)
    isempty(zs) && return xs, (directions = 0, trial_dimension = length(xs))
    if o.search_basis == :direct
        # Equivalent subspace without fitting projected/whitened directions.
        # Resolve linear dependence in the small overlap matrix at Rayleigh–Ritz.
        trial = vcat(xs, zs)
        rec.stage = "rayleigh_ritz"
        ys, eps = spectrum_ritz(
            trial,
            H,
            cfg,
            rec,
            o,
            potential;
            keep = o.states,
            tol = ritz_tolerance,
            reveal_metric = true,
        )
        return ys,
        (
            directions = residual_count,
            trial_dimension = length(trial),
            work_tolerance,
            ritz_tolerance,
            green_tolerance = min(work_tolerance, o.green_tolerance_factor * ritz_tolerance),
            search_basis = "direct",
        )
    end
    raw_overlaps = Q.gram(xs, zs)
    overlaps = Hermitian(Q.gram(xs)) \ raw_overlaps
    ox = Q.block_oracle(xs, rec)
    projected = Q.QTT[]
    rec.stage = "residual_projection"
    for (i, z) in enumerate(zs)
        if i > residual_count
            remaining = sqrt(
                max(
                    0,
                    real(Q.inner(z, z)) - real(dot(view(raw_overlaps, :, i), view(overlaps, :, i))),
                ),
            )
            remaining < o.previous_projection_floor && continue
        end
        ez = Q.evaluate(z, rec)
        q = Q.linear_combination(
            rec,
            vcat([z], xs),
            vcat(1, -view(overlaps, :, i));
            oracle = j -> vcat(ez(j), ox(j)),
            tol = work_tolerance,
            cap = cfg.workcap,
            label = "residual_projection_$i",
        )
        Q.ttnorm(q) > 1e-12 && push!(projected, Q.normalize(q))
    end
    isempty(projected) && return xs, (directions = 0, trial_dimension = length(xs))
    S = Q.gram(projected)
    f = eigen(Hermitian((S + S') / 2))
    good = findall(
        >(max(maximum(f.values) * o.direction_relative_cutoff, o.direction_absolute_cutoff)),
        f.values,
    )
    C = f.vectors[:, good] * Diagonal(1 ./ sqrt.(f.values[good]))
    oracle = Q.block_oracle(projected, rec)
    rec.stage = "residual_basis"
    directions = [
        Q.linear_combination(
            rec,
            projected,
            view(C, :, i);
            oracle,
            tol = work_tolerance,
            cap = cfg.workcap,
            label = "residual_basis_$i",
        ) for i in axes(C, 2)
    ]
    trial = vcat(xs, directions)
    rec.stage = "rayleigh_ritz"
    ys, eps = spectrum_ritz(trial, H, cfg, rec, o, potential; keep = o.states, tol = ritz_tolerance)
    ys,
    (
        directions = length(directions),
        trial_dimension = length(trial),
        trial_gram_defect = norm(Q.gram(trial) - I),
        work_tolerance,
        ritz_tolerance,
    )
end
