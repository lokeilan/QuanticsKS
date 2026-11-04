# Fresh processes keep each suite's module aliases and random seeds independent.
const TEST_PROJECT = dirname(Base.active_project())
for (file, threads) in (
    ("library.jl", "1"),
    ("configuration.jl", "1"),
    ("julia_parallel.jl", "1"),
    ("julia_parallel_extended.jl", "1"),
    ("kernel_optimizations.jl", "4,1"),
    ("julia_parallel.jl", "4,1"),
    ("julia_parallel_extended.jl", "4,1"),
)
    println("Running ", file, " with --threads=", threads)
    flush(stdout)
    run(
        `$(Base.julia_cmd()) --startup-file=no --threads=$threads --project=$TEST_PROJECT $(joinpath(@__DIR__, file))`,
    )
end
println("One-dimensional regression completed.")
