# Replay the first R16 tangent basis to check the removed identity projection.
using QuanticsKS, LinearAlgebra, Random, JSON3, TOML
const K = QuanticsKS
const Q = K.Q
length(ARGS) == 1 || error("Usage: validate_l4.jl EXPERIMENT_DIRECTORY")
out = abspath(only(ARGS))
d = TOML.parsefile(joinpath(out, "inputs", "baseline.toml"))
BLAS.set_num_threads(1)
a = K.options_from_dict(Accuracy, d["accuracy"])
scf = K.options_from_dict(SCFOptions, d["scf"])
tci = K.options_from_dict(TCIOptions, d["tci"])
state = load_checkpoint(joinpath(out, "common-initial", "initial.jls"))
p, xs = state.problem, state.orbitals
cfg = K.solver_config(p, a, scf)
Random.seed!(scf.seed)
rec = K.recorder(nothing, tci; parallel_tasks = scf.parallel_tasks)
try
    ops = K.build_operators(p, a, rec, PhysicsNumerics())
    rec.scf = rec.inner = 1
    n = K.density(p, xs, a, rec; renormalize = true)
    H = K.effective_hamiltonian(p, n, ops, a, rec)
    potential = Q.synchronized(Q.potential_samples(H, rec))
    green = Q.green_mpo(p.grid.R, H.h, scf.green_shift)
    residuals = Q.recorded_map(Q.QTT, length(xs), cfg, rec) do i, localrec
        e = Q.h_expectation(H, xs[i]) / real(Q.inner(xs[i], xs[i]))
        Q.h_residual(
            H,
            xs[i],
            e,
            cfg,
            localrec;
            tol = a.work_tolerance,
            cap = cfg.workcap,
            potential,
            label = "orbital_residual_$i",
        )
    end
    frame = Q.canonical_frame(first(xs); cfg)
    rows = frame.offsets[end]
    DX = Q.independent_columns(i -> Q.project(frame, xs[i]), rows, length(xs), cfg)
    DZ = Q.independent_columns(i -> Q.project(frame, residuals[i], green), rows, length(xs), cfg)
    raw = hcat(DX, DZ)
    for j in axes(raw, 2)
        view(raw, :, j) ./= max(norm(view(raw, :, j)), 1e-14)
    end
    fac = svd!(raw; alg = LinearAlgebra.QRIteration())
    good = findall(
        >(max(maximum(fac.S) * cfg.tangent_relative_cutoff, cfg.tangent_absolute_cutoff)),
        fac.S,
    )
    D = view(fac.U, :, 1:length(good))
    measurements = Any[]
    for j in unique([1, cld(size(D, 2), 2), size(D, 2)])
        z = view(D, :, j)
        t = Q.tangent_tt(frame, z)
        original = Q.h_project(frame, t, H)
        simplified = Q.h_project_tangent(frame, z, H)
        push!(
            measurements,
            (;
                column = j,
                coordinate_identity_defect = norm(Q.project(frame, t) - z),
                projection_absolute_difference = norm(original - simplified),
                projection_relative_difference = norm(original - simplified) / norm(original),
                reduced_column_absolute_difference = norm(D' * (original - simplified)),
            ),
        )
    end
    report = (;
        scope = "Replay of the original-tolerance R16 first-step L1-L4 basis; full original and simplified H projections evaluated on first, middle and last retained SVD columns.",
        tangent_dimension = size(D, 2),
        smallest_retained_singular_value = fac.S[last(good)],
        frame_rank = Q.rank(first(xs)),
        potential_rank = Q.rank(H.potential),
        spacing = H.h,
        measurements,
    )
    open(io -> JSON3.pretty(io, report), joinpath(out, "r16-l4-projection.json"), "w")
    cp(@__FILE__, joinpath(out, basename(@__FILE__)); force = true)
    println(JSON3.write(report))
finally
    close(rec)
end
