# Compare saved outputs on the full grid without solving another SCF step.
using QuanticsKS, LinearAlgebra, JSON3, Statistics
const K = QuanticsKS
const Q = K.Q
length(ARGS) == 1 || error("Usage: compare_scf_outputs.jl EXPERIMENT_DIRECTORY")
const OUT = abspath(only(ARGS))
BLAS.set_num_threads(1)
function values(state)
    Q.independent_columns(
        length(state.problem.grid),
        length(state.orbitals),
        (; parallel_tasks = 0),
    ) do i
        evaluator = Q.Evaluator(state.orbitals[i]; cachebits = 8)
        [evaluator(j) for j = 0:(length(state.problem.grid) - 1)]
    end
end
initial = load_checkpoint(joinpath(OUT, "common-initial", "initial.jls"))
baseline = load_checkpoint(joinpath(OUT, "baseline", "checkpoint.jls"))
X = values(baseline)
h = K.spacing(baseline.problem.grid)
occupations = baseline.problem.occupations
rho(X) = vec(abs2.(X) * occupations) / h
baseline_density = rho(X)
reports = Any[]
for name in ("baseline", "density_xc", "post_svd", "combined")
    println("Comparing full-grid orbitals: ", name)
    flush(stdout)
    state = load_checkpoint(joinpath(OUT, name, "checkpoint.jls"))
    imported = load_checkpoint(joinpath(OUT, name, "initial.jls"))
    all(x.c == y.c for (x, y) in zip(initial.orbitals, imported.orbitals)) ||
        error("Initial cores differ in $name")
    Y = name == "baseline" ? X : values(state)
    raw = [norm(y - x) / norm(x) for (x, y) in zip(eachcol(X), eachcol(Y))]
    aligned = [
        begin
            overlap = dot(x, y)
            phase = iszero(overlap) ? 1.0 + 0im : overlap / abs(overlap)
            norm(y / phase - x) / norm(x)
        end for (x, y) in zip(eachcol(X), eachcol(Y))
    ]
    density = rho(Y)
    overlap_singular_values = svdvals(X' * Y)
    push!(
        reports,
        (;
            case = name,
            identical_initial_cores = true,
            per_orbital_relative_l2 = raw,
            maximum_orbital_relative_l2 = maximum(raw),
            rms_orbital_relative_l2 = sqrt(mean(abs2, raw)),
            per_orbital_phase_aligned_relative_l2 = aligned,
            maximum_phase_aligned_orbital_relative_l2 = maximum(aligned),
            density_relative_l2 = norm(density - baseline_density) / norm(baseline_density),
            electron_number = sum(density) * h,
            full_grid_gram_defect = norm(Y' * Y - I),
            occupied_overlap_singular_values = overlap_singular_values,
            maximum_bond = maximum(Q.rank, state.orbitals),
            orbital_core_bytes = sum(Q.storage, state.orbitals),
            bond_profiles = [[size(g, 3) for g in x.c[1:(end - 1)]] for x in state.orbitals],
        ),
    )
end
report = (;
    scope = "Full-grid comparison against the optimized original-tolerance first-SCF baseline; orbital amplitudes already include sqrt(h). Raw L2 keeps orbital identity and phase; phase-aligned L2 removes only a scalar phase per orbital.",
    R = baseline.problem.grid.R,
    grid_points = length(baseline.problem.grid),
    cases = reports,
)
open(io -> JSON3.pretty(io, report), joinpath(OUT, "full-grid-comparison.json"), "w")
cp(@__FILE__, joinpath(OUT, basename(@__FILE__)); force = true)
