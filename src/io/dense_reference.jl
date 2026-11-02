# Load dense references and align spatial rows, k-point order, and orbital phases.

"Read-only mmap of the admitted native reference; coordinate and scalar-phase map explicit."
function native_reference(root, l; storage = :mmap)
    receipt = TOML.parsefile(joinpath(root, "ground", "completion.toml"))
    get(receipt, "accepted", false) && get(receipt, "complete", false) ||
        error("unadmitted dense reference")
    directory = joinpath(root, "ground", "orbitals")
    m = TOML.parsefile(joinpath(directory, "metadata.toml"))
    m["bits"] == collect(l.space_bits) &&
        m["kmesh"] == collect(kshape(l)) &&
        m["occupied_bands"] == 16 || error("reference grid identity")
    settings = m["model_settings"]
    settings["cell"] == "conventional" &&
        settings["functional"] == "PZ-LDA" &&
        settings["kinetic"] == "bloch_phase_fd" &&
        settings["fd_order"] == 8 &&
        settings["a_bohr"] == l.a_bohr &&
        settings["k_shift"] == collect(l.k_shift) || error("reference physical model identity")
    n = shape(l)
    N = prod(n)
    nk = prod(kshape(l))
    storage in (:mmap, :memory) || error("reference storage must be mmap or memory")
    arrays = Vector{Matrix{ComplexF64}}(undef, 16)
    if storage == :memory
        println("Loading the 32 GiB dense reference into declared working memory")
        flush(stdout)
        Threads.@threads :static for b = 1:16
            arrays[b] = open(joinpath(directory, "orbital-$b.bin"), "r") do io
                read!(io, Matrix{ComplexF64}(undef, N, nk))
            end
        end
    else
        for b = 1:16
            arrays[b] = open(joinpath(directory, "orbital-$b.bin"), "r") do io
                Mmap.mmap(io, Matrix{ComplexF64}, (N, nk); shared = false)
            end
        end
    end
    # Cache spatial rows are x=-a/2,...; return x=0,... for physical operators.
    rows = [
        LinearIndices(n)[ntuple(ax -> mod(I[ax] - 1 + n[ax] ÷ 2, n[ax]) + 1, 3)...] for
        I in vec(collect(CartesianIndices(n)))
    ]
    # Native cache k axes are ascending physical fractional coordinates, not label order.
    ks = kcoordinates(l)
    krows = [
        LinearIndices(kshape(l))[ntuple(
            ax -> mod(k[ax] + kshape(l)[ax] ÷ 2, kshape(l)[ax]) + 1,
            3,
        )...] for k in ks
    ]
    phases = open(joinpath(directory, "phases.bin"), "r") do io
        Mmap.mmap(io, Matrix{ComplexF64}, (nk, 16); shared = false)
    end
    eigenvalues = open(joinpath(directory, "source-eigenvalues.bin"), "r") do io
        Mmap.mmap(io, Matrix{Float64}, (16, nk); shared = false)
    end
    energies = permutedims(eigenvalues[:, krows])
    raw_slice = ik -> hcat([x[rows, krows[ik]] for x in arrays]...)
    slice = ik -> raw_slice(ik) .* transpose(phases[krows[ik], :])
    density = zeros(N)
    rawdensity = read!(joinpath(directory, m["density_file"]), zeros(Float64, N))
    density .= rawdensity # file is already native (unshifted), unlike orbital cache rows
    point = (row, ik, b) -> arrays[b][rows[row], krows[ik]] * phases[krows[ik], b]
    (;
        layout = l,
        slice,
        raw_slice,
        point,
        phases,
        energies,
        density,
        metadata = m,
        receipt,
        root = abspath(root),
        accepted = true,
        electronic_energy = receipt["last_scf"]["electronic_energy"],
    )
end

function reference_slice(reference, ik)
    hasproperty(reference, :slice) ? reference.slice(ik) : reference.orbitals[ik]
end
reference_point(reference, row, ik, b) =
    hasproperty(reference, :point) ? reference.point(row, ik, b) : reference.orbitals[ik][row, b]

function initial_oracle(l, reference, perturbation, b)
    space = LinearIndices(shape(l))
    kgrid = LinearIndices(kshape(l))
    N = prod(shape(l))
    label -> begin
        x, k = decode(l, label)
        row = space[(x .+ 1)...]
        ik = kgrid[(k .+ 1)...]
        reference_point(reference, row, ik, b) +
        perturbation / sqrt(l.a_bohr^3) * cis(2pi * (row * (b + 3) / N + 0.13b))
    end
end

"Independent spatial guess: low-G plane waves plus seeded noise, orthogonal at every k."
