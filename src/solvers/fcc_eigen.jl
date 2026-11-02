# Update four FCC blocks at fixed potential, rebuild QTTs, and check energy changes.

function fcc_ritz(action, trial; keep = 4)
    norms = sqrt.(vec(sum(abs2, trial; dims = 1)))
    good = findall(>(1e-13), norms)
    length(good) >= keep || error("FCC trial has too few nonzero columns")
    qrtrial = qr(trial[:, good] ./ transpose(norms[good]), ColumnNorm())
    diagonal = abs.(diag(qrtrial.R))
    rank = count(>(max(1e-11, maximum(diagonal) * 1e-10)), diagonal)
    rank >= keep || error("FCC trial lost occupied rank")
    basis = Matrix(qrtrial.Q)[:, 1:rank]
    small = basis' * action(basis)
    E = eigen(Hermitian((small + small') / 2))
    Y = basis * E.vectors[:, 1:keep]
    values = E.values[1:keep]
    first = 1
    while first <= keep
        last = first
        while last < keep && abs(values[last + 1] - values[first]) <= 1e-8
            last += 1
        end
        block = first:last
        F = svd(trial[:, block]' * Y[:, block])
        Y[:, block] = Y[:, block] * (F.V * F.U')
        first = last + 1
    end
    Y, values
end

function frozen_inner(p, s::FCCState, potential, o, rec; tol, cap, cache_slices = 8, scf_step = 0)
    old = Inf
    history = Any[]
    energy_stopped = false
    previous = nothing
    for iteration = 1:o["maximum_iterations"]
        factors = spatial_factors(s)
        oldstate = s
        rec.inner = iteration
        rec.stage = "inner"
        previous_factors = previous === nothing ? nothing : spatial_factors(previous)
        callback =
            k -> begin
                kernels = fcc_kernels(p, k, potential)
                [
                    begin
                        action = X -> fcc_action(p, kernels, sector, X)
                        X = sqrt(dv(s.layout)) .* fcc_slice(s, factors, k, sector)
                        X, eps = fcc_ritz(action, X)
                        R = action(X) - X * Diagonal(eps)
                        D = fcc_green(p, kernels, sector, R, o["green_shift"])
                        trial = o["method"] == "block_green" ? X - D : hcat(X, D)
                        if o["method"] == "block_residual" &&
                           previous !== nothing &&
                           get(o, "use_previous_subspace", true)
                            trial = hcat(
                                trial,
                                sqrt(dv(s.layout)) .*
                                fcc_slice(previous, previous_factors, k, sector),
                            )
                        end
                        Y, _ = fcc_ritz(action, trial)
                        Y ./ sqrt(dv(s.layout))
                    end for sector = 1:4
                ]
            end
        s = fit_fcc_slices(
            s.layout,
            callback;
            tol,
            cap,
            rec,
            label = "scf$(scf_step)_inner$iteration",
        )
        GC.gc(true)
        work = s
        prefix = spatial_factors(work)
        pack =
            k -> begin
                kernels = fcc_kernels(p, k, potential)
                [
                    begin
                        X = sqrt(dv(work.layout)) .* fcc_slice(work, prefix, k, sector)
                        Y, _ = fcc_ritz(Y -> fcc_action(p, kernels, sector, Y), X)
                        Y ./ sqrt(dv(work.layout))
                    end for sector = 1:4
                ]
            end
        s = fit_fcc_slices(work.layout, pack; tol, cap, rec, label = "constraint_pack")
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
                maximum_bond = maximum(Q.rank, s.blocks),
            ),
        )
        @printf(
            "INNER FCC %d %d Eband=%.12f dE=%.3e residual=%.3e rank=%d\n",
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
