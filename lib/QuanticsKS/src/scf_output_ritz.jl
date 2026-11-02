# Rotate occupied orbitals with H of the output density, then recompute the density change.

"Canonicalize occupied orbitals using their output-density Hamiltonian."
function canonicalize_output_orbitals(p, xs, ops, a, cfg, rec)
    before = density(p, xs, a, rec; renormalize = true, label = "ritz_output_density")
    Hout = effective_hamiltonian(p, before, ops, a, rec)
    # The complete operator path is used for this final projected matrix.
    # No spatial symmetry reduction or dense reference enters the solver.
    ys, offdiag = Q.ritz_rotate(xs, Hout, cfg, rec; work_tol = 0.0, force = true)
    ys, _ = Q.pack_orbitals(ys, cfg; tol = a.storage_tolerance)
    after = density(p, ys, a, rec; renormalize = true, label = "canonical_output_density")
    density_change = Q.difference_norm(after, before; rec, label = "ritz_density_change") / Q.ttnorm(before)
    ys,
    after,
    (
        final_output_ritz = true,
        final_ritz_offdiagonal_before = offdiag,
        final_ritz_density_relative_change = density_change,
        final_ritz_direction_history_reset = true,
    )
end
