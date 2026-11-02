# Build the four FCC blocks and the transforms needed to recover reference orbitals.

function fcc_reference_frames(l, U, energies)
    frames = Matrix{ComplexF64}[]
    inverses = Matrix{ComplexF64}[]
    for chi in FCC_CHARACTERS
        projected = fcc_restrict_matrix(l, U, chi)
        F = eigen(Hermitian(dv(l) * (projected' * projected)))
        keep = findall(>(0.5), F.values)
        length(keep) == 4 || error("reference FCC occupied sector does not contain four states")
        B = projected * F.vectors[:, keep] * Diagonal(inv.(sqrt.(F.values[keep])))
        C = dv(l) * B' * projected
        norm(B * C - projected) <= 1e-7 * max(norm(projected), eps()) ||
            error("reference FCC projection defect")
        E = eigen(Hermitian(C * Diagonal(energies) * C'))
        push!(frames, B * E.vectors)
        push!(inverses, E.vectors' * C)
    end
    (; frames, inverses)
end
fcc_reference_slice(r, ik) = hasproperty(r, :raw_slice) ? r.raw_slice(ik) : reference_slice(r, ik)

function initialize_fcc(l, c, reference, rec; tol, cap, receipt, progress)
    mode = initialization_mode(c)
    execution = c["execution"]
    isempty(get(execution, "initial_checkpoint", "")) ||
        error("FCC cold/perturbed mode starts independently")
    mode in ("cold", "dense_perturbed") ||
        error("FCC initialization supports cold or dense_perturbed")
    receipt["initialization"] =
        mode == "cold" ? "cold_fcc_plane_wave_random" : "perturbed_dense_fcc_frames"
    receipt["dense_used_for_initialization"] = mode != "cold"
    receipt["initial_perturbation_added_this_run"] =
        mode == "cold" ? 0.0 : get(execution, "initial_perturbation", 0.002)
    receipt["blocks"] = 4
    receipt["bands_per_block"] = 4
    receipt["quotient_shape"] = collect(fcc_shape(l))
    receipt["site_order"] = "x,y,z,kx,ky,kz,band; MSB first"
    receipt["k_label_order"] = "ascending physical fractional axes, aligned with Step1"
    receipt["phase_policy"] = "Step1 conditioned scalar neighbor, largest-sample anchor fallback"
    progress()
    state = if mode == "cold"
        n = shape(l)
        ns = prod(fcc_shape(l))
        N = prod(n)
        allmodes = [
            ntuple(a -> I[a] - 1 < n[a] ÷ 2 ? I[a] - 1 : I[a] - 1 - n[a], 3) for
            I in vec(collect(CartesianIndices(n)))
        ]
        initial = Matrix{ComplexF64}[]
        noise = get(execution, "cold_noise", 0.05)
        seed = get(execution, "cold_seed", 20261031)
        for (sector, chi) in enumerate(FCC_CHARACTERS)
            modes = filter(g -> (-1)^(g[1] + g[3]) == chi[1] && (-1)^(g[2] + g[3]) == chi[2], allmodes)
            sort!(modes; by = g -> (sum(abs2, g), g...))
            modes = modes[1:4]
            trial = ComplexF64[
                2cis(2pi * sum(modes[b][a] * (I[a] - 1) / n[a] for a = 1:3)) / sqrt(N) for
                I in vec(collect(CartesianIndices(fcc_shape(l)))), b = 1:4
            ]
            trial .+=
                noise / sqrt(ns) .* randn(MersenneTwister(seed + sector), ComplexF64, ns, 4)
            push!(initial, Matrix(qr(trial).Q)[:, 1:4] ./ sqrt(dv(l)))
        end
        receipt["cold_seed"] = seed
        receipt["cold_noise"] = noise
        fit_fcc_slices(
            l,
            k -> copy.(initial);
            tol,
            cap,
            rec,
            label = "initial_cold",
            seed = c["interpolation"]["seed"],
        )
    else
        reference === nothing && error("FCC perturbed initialization needs a reference")
        ks = LinearIndices(kshape(l))
        delta = get(execution, "initial_perturbation", 0.002)
        callback =
            k -> begin
                ik = ks[(k .+ 1)...]
                frames =
                    fcc_reference_frames(
                        l,
                        fcc_reference_slice(reference, ik),
                        reference.energies[ik, :],
                    ).frames
                [
                    B .+ ComplexF64[
                        delta / sqrt(prod(fcc_shape(l)) * dv(l)) *
                        cis(2pi * (i * (b + 3) / size(B, 1) + 0.13b)) for i in axes(B, 1),
                        b = 1:4
                    ] for B in frames
                ]
            end
        fit_fcc_slices(
            l,
            callback;
            tol,
            cap,
            rec,
            label = "initial_perturbed",
            seed = c["interpolation"]["seed"],
        )
    end
    receipt["initialized_blocks"] = 4
    receipt["initialized_bands"] = 16
    progress()
    state
end
