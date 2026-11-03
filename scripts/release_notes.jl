# Generate publication notes from the recorded V1 numerical measurements.
using JSON3, Printf
root = normpath(joinpath(@__DIR__, ".."))
d = JSON3.read(read(joinpath(root, "docs", "releases", "v1.json"), String))
fmt(x) = @sprintf("%.10g", x)
open(joinpath(root, "RELEASE_NOTES.md"), "w") do io
    println(io, "# V1 (dedup) — ", d.version, "\n")
    println(io, "Publishes registered version `", d.registered_version_id, "`. Numerical source commit: `", d.numerical_source_commit, "`, also tagged `", d.numerical_source_tag, "`. Publication documentation is added on top of this numerical snapshot.\n")
    println(io, "V1 reduces repeated sampling setup and Gram contractions, shares read-only pivot snapshots and fixed-H potential samples, and retains pivot warm starts, sampled Ritz projection and combined orbital reconstruction.\n")
    println(io, "## Recorded R16 cold start\n")
    println(io, "Model: ", d.model, ". The grid has ", d.grid_points, " points and ", d.occupied_orbitals, " doubly occupied orbitals. The recorded run used Julia ", d.julia_version, ", ", d.julia_threads, " Julia threads and ", d.blas_threads, " BLAS threads.\n")
    println(io, "| Measurement | Recorded value |\n|---|---:|")
    for (label, value) in (("SCF iterations", d.scf_iterations),
                           ("Energy-change stopping tolerance (Ha)", d.energy_stopping_tolerance_Ha),
                           ("Final energy change (Ha)", d.final_energy_change_Ha),
                           ("Final total energy (Ha)", d.final_total_energy_Ha),
                           ("Maximum stored orbital rank", d.max_orbital_rank),
                           ("SCF iteration time sum (s)", d.scf_seconds),
                           ("Complete Julia process wall time (s)", d.process_wall_seconds),
                           ("Audit maximum single-orbital residual (Ha)", d.audit_max_single_orbital_residual_Ha),
                           ("Audit relative density residual", d.audit_density_relative_residual),
                           ("Audit Gram defect", d.audit_gram_defect))
        println(io, "| ", label, " | ", fmt(value), " |")
    end
    println(io, "\nThe energy-change criterion was reached. Independent physical audit passed: **", d.independent_audit_passed, "**. The state is not certified as physically converged.\n")
    println(io, "SCF time is the sum of iteration timers. Complete-process time also includes startup, initialization, output and final audit. These timings describe one archived local run.\n")
    println(io, "## Verification and scope\n")
    println(io, "The V1 library regressions and forced fixed-H Ritz validation completed in the recorded campaign. The publication checkout is separately checked for package loading, R16 input planning and library regression. These checks do not replace physical acceptance.\n")
    println(io, d.limitations, "\n")
    println(io, "The release contains source, tests, pinned dependencies, the R16 input and publication documentation. Archived outputs, checkpoints and later working-tree changes are excluded. GitHub supplies the source download for this tag.\n")
    println(io, "Recorded measurement provenance: ", d.evidence, ".")
end
