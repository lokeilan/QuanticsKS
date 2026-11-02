# Fit joint x-k-band QTTs for four FCC blocks and record TCI error estimates and storage.

# Direct CachedFunction -> crossinterpolate2 -> QTT, as in FCC Step1.
function fcc_cross(rec, l, fn; tol, cap, sector, label, seed = 20261029)
    dims = fill(2, fcc_sites(l))
    Random.seed!(seed + sector)
    cached = TCI.CachedFunction{ComplexF64}(q -> fn(Q.indexof(q)), dims)
    rec.calls += 1
    started = time_ns()
    ci, ranks, errors = TCI.crossinterpolate2(
        ComplexF64,
        cached,
        dims,
        [ones(Int, length(dims))];
        tolerance = tol,
        maxbonddim = cap,
        maxiter = rec.options.max_iterations,
        nsearchglobalpivot = rec.options.global_pivot_search,
    )
    seconds = (time_ns() - started) / 1e9
    rec.tci_time += seconds
    rec.samples += length(cached.cache)
    train = Q.QTT(TCI.TensorTrain(ci).sitetensors)
    Q.event!(
        rec.tci_io,
        (;
            call = rec.calls,
            scf = rec.scf,
            inner = rec.inner,
            stage = rec.stage,
            label,
            seconds,
            tolerance = tol,
            rank_cap = cap,
            rank = Q.rank(train),
            sweeps = length(ranks),
            reported_error = isempty(errors) ? 0.0 : last(errors),
            reported_converged = !isempty(errors) && last(errors) <= tol,
            at_rank_cap = Q.rank(train) >= cap,
            oracle_evaluations = length(cached.cache),
            scalar_cache_bytes = Base.summarysize(cached.cache),
            post_rounding = false,
            construction = "Step1 direct CachedFunction TCI",
            seed = seed + sector,
        ),
    )
    rec.options.require_convergence &&
        (isempty(errors) || last(errors) > tol) &&
        error("FCC TCI target not met")
    train
end

function fit_fcc_slices(l, callback; tol, cap, rec, label = "FCC", seed = 20261029)
    # Tensor k labels follow Step1's sorted physical axes; public k indices
    # retain JointLayout's convention for the physical Hamiltonian backend.
    ks = [fcc_klabel(l, Tuple(I) .- 1) for I in vec(collect(CartesianIndices(kshape(l))))]
    ns = prod(fcc_shape(l))
    nk = length(ks)
    li = LinearIndices(kshape(l))
    blocks = Vector{Vector{Matrix{ComplexF64}}}(undef, nk)
    println("FCC ", label, " workspace: ", 16ns * 16nk, " bytes for four quotient blocks")
    flush(stdout)
    Q.timed(rec, "fcc_slice_workspace"; item = label) do
        Threads.@threads :static for ik in eachindex(ks)
            value = callback(ks[ik])
            length(value) == 4 && all(x -> size(x) == (ns, 4), value) ||
                error("FCC callback dimensions")
            blocks[ik] = value
        end
    end
    phases = ones(ComplexF64, nk, 4, 4)
    # Identical predecessor and scalar-only transport rule to Step1 fcc_prepare.
    for ik = 1:nk
        ki = Tuple(CartesianIndices(kshape(l))[ik])
        parent = 0
        for a = 1:3
            if ki[a] > 1
                index = ntuple(j -> j == a ? ki[j] - 1 : ki[j], 3)
                parent = li[index...]
                break
            end
        end
        for sector = 1:4, b = 1:4
            B = blocks[ik][sector]
            z =
                parent == 0 ? 0.0im :
                conj(phases[parent, b, sector]) *
                dv(l) *
                dot(view(blocks[parent][sector], :, b), view(B, :, b))
            if abs(z) < 1e-4
                z = B[argmax(abs.(view(B, :, b))), b]
                abs(z) > 0 || error("zero FCC block orbital")
            end
            phases[ik, b, sector] = conj(z) / abs(z)
        end
    end
    space = LinearIndices(fcc_shape(l))
    trains = Q.QTT[]
    for sector = 1:4
        println("TCI ", label, " FCC block ", sector, "/4")
        flush(stdout)
        oracle = j -> begin
            x, k, b = fcc_decode(l, j)
            ik = fcc_kindex(l, k)
            blocks[ik][sector][space[(x .+ 1)...], b] * phases[ik, b, sector]
        end
        push!(
            trains,
            fcc_cross(rec, l, oracle; tol, cap, sector, label = "$(label)_block$sector", seed),
        )
        if rec.dir !== nothing && startswith(label, "initial_")
            serialize(
                joinpath(rec.dir, "$label-block-$sector.jls"),
                (;
                    train = last(trains),
                    sector,
                    phases = phases[:, :, sector],
                    layout = l,
                    tolerance = tol,
                    cap,
                ),
            )
        end
        rec.dir === nothing || !isfile(joinpath(rec.dir, "STOP")) || throw(InterruptException())
        GC.gc()
    end
    FCCState(l, trains, phases)
end
