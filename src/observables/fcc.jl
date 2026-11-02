# FCC density and reference errors; restore phases and reference column transforms first.

function occupied_density(p, s::FCCState; occupied = 16, factors = spatial_factors(s))
    occupied == 16 || error("FCC density uses four occupied bands per sector")
    ks = kcoordinates(s.layout)
    n = fcc_shape(s.layout)
    buffers = [zeros(prod(n)) for _ = 1:Threads.maxthreadid()]
    Threads.@threads :static for ik in eachindex(ks)
        target = buffers[Threads.threadid()]
        for sector = 1:4
            V = fcc_slice(s, factors, ks[ik], sector; raw = false)
            target .+= 0.5 / length(ks) .* vec(sum(abs2, V; dims = 2))
        end
    end
    # Each expanded orbital has amplitude 1/2 on four translated copies.
    real.(2vec(fcc_expand(reshape(reduce(+, buffers), n), (1, 1))))
end

function compare_dense(p, s::FCCState, reference, out)
    l = s.layout
    ks = kcoordinates(l)
    factors = spatial_factors(s)
    w = 1 / length(ks)
    raw = zeros(16)
    phase = zeros(16)
    degenerate = zeros(16)
    den = zeros(16)
    worst = zeros(16)
    blockraw = zeros(4, 4)
    blockphase = zeros(4, 4)
    blockden = zeros(4, 4)
    sub = zeros(length(ks))
    de = zeros(length(ks), 16)
    rho = zeros(prod(shape(l)))
    fields = physical_fields(p, reference.density)
    inverses = Array{ComplexF64}(undef, 4, 16, length(ks), 4)
    for (ik, k) in enumerate(ks)
        D = fcc_reference_slice(reference, ik)
        ref = fcc_reference_frames(l, D, reference.energies[ik, :])
        original = zeros(ComplexF64, size(D))
        for sector = 1:4
            V = fcc_slice(s, factors, k, sector)
            B = ref.frames[sector]
            C = ref.inverses[sector]
            inverses[:, :, fcc_kindex(l, k), sector] = C
            original .+= fcc_expand_matrix(l, V * C, FCC_CHARACTERS[sector])
            for b = 1:4
                U = sqrt(dv(l)) .* V[:, b]
                Y = sqrt(dv(l)) .* B[:, b]
                blockraw[b, sector] += w * sum(abs2, U - Y)
                blockphase[b, sector] += w * sum(abs2, U * cis(-angle(dot(Y, U))) - Y)
                blockden[b, sector] += w * sum(abs2, Y)
            end
        end
        X = sqrt(dv(l)) .* original
        Y = sqrt(dv(l)) .* D
        aligned = copy(X)
        first = 1
        while first <= 16
            last = first
            while last < 16 &&
                abs(reference.energies[ik, last + 1] - reference.energies[ik, first]) <= 1e-8
                last += 1
            end
            block = first:last
            F = svd(X[:, block]' * Y[:, block])
            aligned[:, block] = X[:, block] * (F.U * F.Vt)
            first = last + 1
        end
        for b = 1:16
            er = sum(abs2, X[:, b] - Y[:, b])
            nr = sum(abs2, Y[:, b])
            raw[b] += w * er
            den[b] += w * nr
            worst[b] = max(worst[b], sqrt(er / nr))
            phase[b] += w * sum(abs2, X[:, b] * cis(-angle(dot(Y[:, b], X[:, b]))) - Y[:, b])
            degenerate[b] += w * sum(abs2, aligned[:, b] - Y[:, b])
        end
        gx = X' * X
        gy = Y' * Y
        sub[ik] = sqrt(max(0.0, real(sum(abs2, gx) + sum(abs2, gy) - 2sum(abs2, X' * Y))) / 32)
        HX = haction(p, k, X, fields.potential)
        HY = haction(p, k, Y, fields.potential)
        de[ik, :] =
            abs.(real.(diag(X' * HX)) ./ real.(diag(gx)) .- real.(diag(Y' * HY)) ./ real.(diag(gy)))
        rho .+= 2w .* vec(sum(abs2, original; dims = 2))
    end
    storage = storage_report(s)
    serialize(
        joinpath(out, "reference-fcc-inverse.jls"),
        (;
            inverses,
            reference_gauge = "raw native original columns",
            k_label_order = "ascending physical axes",
            characters = FCC_CHARACTERS,
        ),
    )
    open(joinpath(out, "orbital-errors.csv"), "w") do io
        println(
            io,
            "orbital,raw_relative_l2,phase_aligned_relative_l2,worst_k_raw_relative_l2,fixed_H_energy_max_error_Ha,fixed_H_energy_rms_error_Ha,achieved_maximum_bond,core_bytes,degenerate_block_aligned_relative_l2,storage_scope",
        )
        for b = 1:16
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
                        storage.maximum_bond,
                        storage.core_bytes,
                        sqrt(degenerate[b] / den[b]),
                        "shared_four_FCC_trains",
                    ),
                    ',',
                ),
            )
        end
    end
    open(joinpath(out, "block-orbital-errors.csv"), "w") do io
        println(
            io,
            "block,band,raw_relative_l2,phase_aligned_relative_l2,train_maximum_bond,train_core_bytes",
        )
        for sector = 1:4, b = 1:4
            println(
                io,
                join(
                    (
                        sector,
                        b,
                        sqrt(blockraw[b, sector] / blockden[b, sector]),
                        sqrt(blockphase[b, sector] / blockden[b, sector]),
                        Q.rank(s.blocks[sector]),
                        Q.storage(s.blocks[sector]),
                    ),
                    ',',
                ),
            )
        end
    end
    open(joinpath(out, "bond-profile.csv"), "w") do io
        println(io, "orbital,bond,rank,sector")
        nx = sum(fcc_bits(l))
        nk = sum(l.k_bits)
        for (b, x) in enumerate(s.blocks), bond = 1:(length(x) - 1)
            sector =
                bond < nx ? "space" :
                bond == nx ? "space_k" : bond < nx + nk ? "k" : bond == nx + nk ? "k_band" : "band"
            println(io, "$b,$bond,$(size(x.c[bond],3)),$sector")
        end
    end
    (;
        aggregate_raw_relative_l2 = sqrt(sum(raw) / sum(den)),
        maximum_orbital_raw_relative_l2 = maximum(sqrt.(raw ./ den)),
        aggregate_phase_aligned_relative_l2 = sqrt(sum(phase) / sum(den)),
        aggregate_degenerate_block_aligned_relative_l2 = sqrt(sum(degenerate) / sum(den)),
        degenerate_resolution_Ha = 1e-8,
        aggregate_block_raw_relative_l2 = sqrt(sum(blockraw) / sum(blockden)),
        maximum_k_subspace = maximum(sub),
        density_relative_l2 = norm(rho - reference.density) / norm(reference.density),
        orbital_bytes = storage.core_bytes,
        maximum_bond = storage.maximum_bond,
        phase_bytes = sizeof(s.phases),
        inverse_bytes = sizeof(inverses),
        retained_bytes = storage.core_bytes + sizeof(s.phases) + sizeof(inverses),
        reference_gauge = "raw native original columns",
        representation = "four FCC joint x-k-band trains",
    )
end
