# Rebuild H from the output density; report maximum and RMS residuals across all k points.

function occupied_density(p, s; occupied = 16, factors = spatial_factors(s))
    ks = kcoordinates(s.layout)
    nt = Threads.maxthreadid()
    buffers = [zeros(length(p.local_potential)) for _ = 1:nt]
    Threads.@threads :static for ik in eachindex(ks)
        X = state_slice(s, factors, ks[ik])[:, 1:occupied]
        buffers[Threads.threadid()] .+= 2 / length(ks) .* vec(sum(abs2, X; dims = 2))
    end
    reduce(+, buffers)
end

"Fresh full-grid evaluation, including all k and raw charge/Gram, without repairs."
function measure_state(p, s; occupied = 16, potential = nothing)
    l = s.layout
    ks = kcoordinates(l)
    factors = spatial_factors(s)
    rho = occupied_density(p, s; occupied, factors)
    fields = physical_fields(p, rho)
    v = potential === nothing ? fields.potential : potential
    nt = Threads.maxthreadid()
    ekin = zeros(nt)
    enl = zeros(nt)
    grams = zeros(length(ks))
    residuals = zeros(length(ks))
    maxcolumns = zeros(length(ks))
    band = zeros(length(ks), occupied)
    Threads.@threads :static for ik in eachindex(ks)
        k = ks[ik]
        X = sqrt(dv(l)) .* state_slice(s, factors, k)[:, 1:occupied]
        action = haction(p, k, X, v; parts = true)
        A = X' * action.HX
        S = X' * X
        R = action.HX - X * A
        grams[ik] = norm(S - I)
        residuals[ik] = norm(R)
        maxcolumns[ik] = maximum(sqrt.(vec(sum(abs2, R; dims = 1))))
        band[ik, :] = real.(diag(A))
        tid = Threads.threadid()
        ekin[tid] += 2 / length(ks) * real(sum(conj.(X) .* action.TX))
        enl[tid] += 2 / length(ks) * real(sum(conj.(X) .* action.NL))
    end
    external = dv(l) * dot(rho, p.local_potential)
    eh = 0.5dv(l) * dot(rho, fields.hartree)
    electronic = sum(ekin) + sum(enl) + external + eh + fields.xc_energy
    (;
        density = rho,
        potential = fields.potential,
        band_energies = band,
        band_energy = 2sum(band) / length(ks),
        electronic_energy = electronic,
        total_energy = electronic + p.ion_energy,
        ion_energy = p.ion_energy,
        kinetic = sum(ekin),
        nonlocal = sum(enl),
        external,
        hartree = eh,
        xc = fields.xc_energy,
        charge = dv(l) * sum(rho),
        maximum_k_gram = maximum(grams),
        maximum_k_residual = maximum(residuals),
        rms_k_residual = sqrt(sum(abs2, residuals) / length(ks)),
        maximum_orbital_residual = maximum(maxcolumns),
        worst_k = argmax(residuals),
        residuals,
        grams,
    )
end

function physical_admitted(measure, density_residual, validation)
    density_required = get(validation, "require_density_selfconsistency", true)
    measure.maximum_k_gram <= validation["maximum_k_gram_F"] &&
        abs(measure.charge - 32) <= validation["charge_error_electrons"] &&
        measure.maximum_k_residual <= validation["maximum_k_residual_F_Ha"] &&
        (!density_required || density_residual <= validation["density_selfconsistency_relative_L2"])
end
"Raw and phase-only orbital L2, occupied-subspace errors, and fixed-H energy errors."
function compare_dense(p, s, reference, out)
    l = s.layout
    ks = kcoordinates(l)
    fac = spatial_factors(s)
    nocc = 16
    raw = zeros(nocc)
    phase = zeros(nocc)
    degenerate = zeros(nocc)
    den = zeros(nocc)
    worst = zeros(nocc)
    sub = zeros(length(ks))
    de = zeros(length(ks), nocc)
    aggregate = 0.0
    refaggregate = 0.0
    rho = zeros(prod(shape(l)))
    reffields = physical_fields(p, reference.density)
    for (ik, k) in enumerate(ks)
        U = state_slice(s, fac, k)[:, 1:nocc]
        D = reference_slice(reference, ik)
        w = 1 / length(ks)
        X = sqrt(dv(l)) .* U
        Y = sqrt(dv(l)) .* D
        aligned = copy(X)
        first = 1
        while first <= nocc
            last = first
            while last < nocc &&
                abs(reference.energies[ik, last + 1] - reference.energies[ik, first]) <= 1e-8
                last += 1
            end
            block = first:last
            F = svd(X[:, block]' * Y[:, block])
            aligned[:, block] = X[:, block] * (F.U * F.Vt)
            first = last + 1
        end
        for b = 1:nocc
            nr = norm(Y[:, b])^2
            er = norm(X[:, b] - Y[:, b])^2
            overlap = dot(Y[:, b], X[:, b])
            phase_error = norm(X[:, b] * cis(-angle(overlap)) - Y[:, b])^2
            raw[b] += w * er
            phase[b] += w * phase_error
            degenerate[b] += w * norm(aligned[:, b] - Y[:, b])^2
            den[b] += w * nr
            worst[b] = max(worst[b], sqrt(er / nr))
        end
        gramX = X' * X
        gramY = Y' * Y
        sub[ik] =
            sqrt(max(0, real(sum(abs2, gramX) + sum(abs2, gramY) - 2sum(abs2, X' * Y))) / (2nocc))
        HX = haction(p, k, X, reffields.potential)
        HY = haction(p, k, Y, reffields.potential)
        de[ik, :] =
            abs.(
                real.(diag(X' * HX)) ./ real.(diag(gramX)) .-
                real.(diag(Y' * HY)) ./ real.(diag(gramY))
            )
        rho .+= 2w .* vec(sum(abs2, U; dims = 2))
    end
    open(joinpath(out, "orbital-errors.csv"), "w") do io
        println(
            io,
            "orbital,raw_relative_l2,phase_aligned_relative_l2,worst_k_raw_relative_l2,fixed_H_energy_max_error_Ha,fixed_H_energy_rms_error_Ha,achieved_maximum_bond,core_bytes,degenerate_block_aligned_relative_l2",
        )
        for b = 1:nocc
            println(
                io,
                join(
                    (
                        b,
                        sqrt(raw[b] / den[b]),
                        sqrt(phase[b] / den[b]),
                        worst[b],
                        maximum(de[:, b]),
                        sqrt(sum(abs2, de[:, b]) / length(ks)),
                        Q.rank(s.orbitals[b]),
                        Q.storage(s.orbitals[b]),
                        sqrt(degenerate[b] / den[b]),
                    ),
                    ',',
                ),
            )
        end
    end
    open(joinpath(out, "bond-profile.csv"), "w") do io
        println(io, "orbital,bond,rank,sector")
        for (b, x) in enumerate(s.orbitals), bond = 1:(length(x) - 1)
            sector =
                bond < sum(l.space_bits) ? "space" : bond == sum(l.space_bits) ? "space_k" : "k"
            println(io, "$b,$bond,$(size(x.c[bond],3)),$sector")
        end
    end
    (;
        aggregate_raw_relative_l2 = sqrt(sum(raw) / sum(den)),
        maximum_orbital_raw_relative_l2 = maximum(sqrt.(raw ./ den)),
        aggregate_phase_aligned_relative_l2 = sqrt(sum(phase) / sum(den)),
        aggregate_degenerate_block_aligned_relative_l2 = sqrt(sum(degenerate) / sum(den)),
        degenerate_resolution_Ha = 1e-8,
        maximum_k_subspace = maximum(sub),
        density_relative_l2 = norm(rho - reference.density) / norm(reference.density),
        orbital_bytes = sum(Q.storage, s.orbitals),
        maximum_bond = maximum(Q.rank, s.orbitals),
    )
end
