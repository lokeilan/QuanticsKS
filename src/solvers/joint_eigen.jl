# Update each k slice at fixed potential and rebuild joint QTTs; density mixing is outside.

"Rank-revealing physical trial space, followed by a per-k Rayleigh-Ritz solve."
function ritz_block(p, k, trial, potential, keep)
    norms = sqrt.(vec(sum(abs2, trial; dims = 1)))
    good = findall(>(1e-13), norms)
    length(good) >= keep || error("insufficient nonzero trial columns")
    Y = trial[:, good] ./ transpose(norms[good])
    f = qr(Y, ColumnNorm())
    diagonal = abs.(diag(f.R))
    r = count(>(max(1e-11, maximum(diagonal) * 1e-10)), diagonal)
    r >= keep || error("per-k trial space lost occupied rank")
    basis = Matrix(f.Q)[:, 1:r]
    HB = haction(p, k, basis, potential)
    small = basis' * HB
    F = eigen(Hermitian((small + small') / 2))
    result = basis * F.vectors[:, 1:keep]
    values = F.values[1:keep]
    # Align degenerate blocks with the previous orbitals to limit arbitrary rotations in the TCI fit.
    anchor = trial[:, 1:keep]
    first = 1
    while first <= keep
        last = first
        while last < keep && abs(values[last + 1] - values[first]) <= 1e-8
            last += 1
        end
        block = first:last
        overlap = anchor[:, block]' * result[:, block]
        alignment = svd(overlap)
        result[:, block] = result[:, block] * (alignment.V * alignment.U')
        first = last + 1
    end
    result, values
end

function frozen_inner(p, s, potential, o, rec; tol, cap, cache_slices = 8, scf_step = 0)
    keep = length(s.orbitals)
    old = Inf
    history = Any[]
    energy_stopped = false
    previous = nothing
    for iteration = 1:o["maximum_iterations"]
        factors = spatial_factors(s)
        rec.inner = iteration
        rec.stage = "inner"
        oldstate = s
        previous_factors = previous === nothing ? nothing : spatial_factors(previous)
        callback =
            k -> begin
                # Multiplying physical samples by sqrt(dV) gives the Euclidean constraint X'X=I.
                X = sqrt(dv(s.layout)) .* state_slice(s, factors, k)
                X, eps = ritz_block(p, k, X, potential, keep)
                HX = haction(p, k, X, potential)
                R = HX - X * Diagonal(eps)
                direction = green_action(p, k, R, o["green_shift"])
                trial = o["method"] == "block_green" ? X - direction : hcat(X, direction)
                if o["method"] == "block_residual" &&
                   previous !== nothing &&
                   get(o, "use_previous_subspace", true)
                    trial = hcat(
                        trial,
                        sqrt(dv(s.layout)) .* state_slice(previous, previous_factors, k),
                    )
                end
                Y, _ = ritz_block(p, k, trial, potential, keep)
                Y ./ sqrt(dv(s.layout))
            end
        s = fit_slices(
            s.layout,
            keep,
            callback;
            tol,
            cap,
            rec,
            cache_slices,
            label = "scf$(scf_step)_inner$iteration",
        )
        GC.gc(true) # release the completed operation's bounded slice cache
        # Constraint reconstruction has a separate accuracy budget from the work update.
        work = s
        prefix = spatial_factors(work)
        pack = k -> begin
            X = sqrt(dv(work.layout)) .* state_slice(work, prefix, k)
            Y, _ = ritz_block(p, k, X, potential, keep)
            Y ./ sqrt(dv(work.layout))
        end
        s = fit_slices(
            work.layout,
            keep,
            pack;
            tol,
            cap,
            rec,
            cache_slices,
            label = "constraint_pack",
        )
        GC.gc(true)
        previous = oldstate
        if rec.dir !== nothing
            checkpoint = joinpath(rec.dir, "inner-checkpoint.jls")
            serialize(checkpoint * ".tmp", (; state = s, scf_step, iteration))
            mv(checkpoint * ".tmp", checkpoint; force = true)
        end
        measure = measure_state(p, s; potential)
        change = abs(measure.band_energy - old)
        push!(
            history,
            (;
                iteration,
                band_energy = measure.band_energy,
                energy_change = change,
                maximum_k_gram = measure.maximum_k_gram,
                maximum_k_residual = measure.maximum_k_residual,
                maximum_bond = maximum(Q.rank, s.orbitals),
            ),
        )
        @printf(
            "INNER %d %d Eband=%.12f dE=%.3e residual=%.3e rank=%d\n",
            scf_step,
            iteration,
            measure.band_energy,
            change,
            measure.maximum_k_residual,
            last(history).maximum_bond
        )
        flush(stdout)
        if iteration >= o["minimum_iterations"] && change <= o["energy_tolerance_Ha"]
            energy_stopped = true
            break
        end
        old = measure.band_energy
    end
    (; state = s, history, energy_stopped)
end
