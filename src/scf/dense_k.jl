# Dense k-point reference for small spatial grids, bounded by maximum_points.

"Independent dense direct diagonalization, bounded to a declared small grid."
function dense_scf(
    p;
    max_steps = 150,
    mixing = 0.2,
    density_tolerance = 1e-9,
    residual_tolerance = 1e-8,
    energy_tolerance = 1e-10,
    maximum_points = 512,
)
    l = p.layout
    N = prod(shape(l))
    N <= maximum_points || error("dense direct matrix point budget exceeded")
    ks = kcoordinates(l)
    rho = initial_density(p)
    old = Inf
    history = Any[]
    inputs = Vector{Float64}[]
    errors = Vector{Float64}[]
    orbitals = Vector{Matrix{ComplexF64}}(undef, length(ks))
    accepted = false
    final = nothing
    for step = 1:max_steps
        fields = physical_fields(p, rho)
        out = zeros(N)
        e0 = 0.0
        energies = zeros(length(ks), 16)
        for (ik, k) in enumerate(ks)
            F = eigen(dense_hamiltonian(p, k, fields.potential), 1:16)
            X = F.vectors
            orbitals[ik] = X ./ sqrt(dv(l))
            energies[ik, :] = F.values
            out .+= 2 / length(ks) / dv(l) .* vec(sum(abs2, X; dims = 2))
            action = haction(p, k, X, zeros(N); parts = true)
            e0 += 2 / length(ks) * real(sum(conj.(X) .* (action.TX + action.NL)))
        end
        fresh = physical_fields(p, out)
        energy =
            e0 +
            dv(l) * dot(out, p.local_potential) +
            0.5dv(l) * dot(out, fresh.hartree) +
            fresh.xc_energy
        residual = 0.0
        gram = 0.0
        for (ik, k) in enumerate(ks)
            X = sqrt(dv(l)) .* orbitals[ik]
            HX = haction(p, k, X, fresh.potential)
            residual = max(residual, norm(HX - X * (X' * HX)))
            gram = max(gram, norm(X' * X - I))
        end
        drho = norm(out - rho) / norm(out)
        change = abs(energy - old)
        row = (;
            step,
            electronic_energy = energy,
            total_energy = energy + p.ion_energy,
            energy_change = change,
            density_residual = drho,
            maximum_k_residual = residual,
            maximum_k_gram = gram,
            charge = dv(l) * sum(out),
        )
        push!(history, row)
        @printf(
            "DENSE %d E=%.12f dE=%.3e drho=%.3e residual=%.3e\n",
            step,
            energy,
            change,
            drho,
            residual
        )
        flush(stdout)
        final = (; density = out, energy, energies)
        if step >= 2 &&
           change <= energy_tolerance &&
           drho <= density_tolerance &&
           residual <= residual_tolerance
            accepted = true
            break
        end
        rho = density_mix(rho, out, inputs, errors; mixing)
        old = energy
    end
    (;
        layout = l,
        orbitals,
        density = final.density,
        electronic_energy = final.energy,
        total_energy = final.energy + p.ion_energy,
        energies = final.energies,
        history,
        accepted,
    )
end
