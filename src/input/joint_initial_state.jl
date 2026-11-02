# Initialize joint orbitals from a cold start, perturbed reference, or checkpoint.

function cold_spatial_guess(l; bands = 16, seed = 20261031, noise = 0.05)
    n = shape(l)
    N = prod(n)
    0 < bands <= N && isfinite(noise) && noise >= 0 || error("invalid cold guess controls")
    modes = [
        ntuple(a -> I[a] - 1 < n[a] ÷ 2 ? I[a] - 1 : I[a] - 1 - n[a], 3) for
        I in vec(collect(CartesianIndices(n)))
    ]
    sort!(modes; by = g -> (sum(abs2, g), g...))
    selected = modes[1:bands]
    points = vec(collect(CartesianIndices(n)))
    trial = ComplexF64[
        cis(2pi * sum(selected[b][a] * (I[a] - 1) / n[a] for a = 1:3)) / sqrt(N) for
        I in points, b = 1:bands
    ]
    rng = MersenneTwister(seed)
    trial .+= noise / sqrt(N) .* randn(rng, ComplexF64, N, bands)
    # This spatial QR defines the initial guess only. Since it is independent of k,
    # it is also the physical per-k QR; no k-averaged constraint is substituted.
    factors = qr(trial)
    X = Matrix(factors.Q)[:, 1:bands]
    (;
        values = X ./ sqrt(dv(l)),
        modes = selected,
        seed,
        noise,
        workspace_bytes = sizeof(trial) + sizeof(X),
    )
end

initialization_mode(c) = get(c["execution"], "initialization", "dense_perturbed")

"Build a full joint initial state; cold mode cannot use a reference or checkpoint."
function initialize_joint(
    l,
    c,
    reference,
    rec;
    tol,
    cap,
    receipt = Dict{String,Any}(),
    progress = () -> nothing,
)
    get(get(c, "representation", Dict()), "kind", "joint_xk") == "fcc_joint_xk_band" &&
        return initialize_fcc(l, c, reference, rec; tol, cap, receipt, progress)
    execution = c["execution"]
    mode = initialization_mode(c)
    mode in ("dense_perturbed", "cold", "checkpoint") || error("unknown initialization: $mode")
    warm_path = get(execution, "initial_checkpoint", "")
    mode == "checkpoint" ||
        isempty(warm_path) ||
        error("initial_checkpoint requires explicit checkpoint initialization")
    if mode == "checkpoint"
        isempty(warm_path) && error("checkpoint initialization needs initial_checkpoint")
        loaded = deserialize(warm_path)
        state = loaded isa JointState ? loaded : loaded.state
        old = state.layout
        old.space_bits == l.space_bits &&
            old.k_bits == l.k_bits &&
            old.a_bohr == l.a_bohr &&
            old.k_shift == l.k_shift || error("warm checkpoint layout identity")
        length(state.orbitals) == 16 && maximum(Q.rank, state.orbitals) <= cap ||
            error("warm checkpoint band count or cap")
        receipt["initialization"] = "warm_checkpoint"
        receipt["initial_checkpoint"] = warm_path
        receipt["initial_perturbation_added_this_run"] = 0.0
        return state
    end
    perturbation = mode == "dense_perturbed" ? get(execution, "initial_perturbation", 0.002) : 0.0
    mode == "dense_perturbed" &&
        reference === nothing &&
        error("dense_perturbed initialization needs a reference")
    cold =
        mode == "cold" ?
        cold_spatial_guess(
            l;
            seed = get(execution, "cold_seed", 20261031),
            noise = get(execution, "cold_noise", 0.05),
        ) : nothing
    receipt["initialization"] =
        mode == "cold" ? "cold_plane_wave_random" : "perturbed_dense_entry_oracle"
    receipt["initial_perturbation_added_this_run"] = perturbation
    if mode == "cold"
        receipt["cold_seed"] = cold.seed
        receipt["cold_noise"] = cold.noise
        receipt["cold_modes"] = collect.(cold.modes)
        receipt["cold_workspace_bytes"] = cold.workspace_bytes
        receipt["dense_used_for_initialization"] = false
    else
        receipt["dense_used_for_initialization"] = true
    end
    orbitals = Q.QTT[]
    space = LinearIndices(shape(l))
    for b = 1:16
        println("TCI ", mode, " initial orbital $b/16")
        flush(stdout)
        oracle =
            mode == "cold" ? (label -> cold.values[space[(decode(l, label)[1] .+ 1)...], b]) :
            initial_oracle(l, reference, perturbation, b)
        push!(orbitals, Q.cross(rec, sites(l), oracle; tol, cap, label = "initial_$(mode)_$b"))
        rec.dir === nothing || serialize(
            joinpath(rec.dir, "initial-orbital-$b.jls"),
            (; orbital = last(orbitals), layout = l, tolerance = tol, cap, initialization = mode),
        )
        receipt["initialized_bands"] = b
        progress()
        rec.dir === nothing || !isfile(joinpath(rec.dir, "STOP")) || throw(InterruptException())
    end
    JointState(l, orbitals)
end
