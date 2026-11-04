using Test, QuanticsKS, JSON3, TOML, LinearAlgebra
BLAS.set_num_threads(1)

@testset "Configured SCF restart restores pivot geometry and starts fresh history" begin
    mktempdir() do dir
        settings = Dict(
            "run" => Dict("kind" => "scf", "blas_threads" => 1),
            "grid" => Dict("R" => 5, "cell" => 8.0),
            "system" => Dict("kind" => "hydrogen_chain", "natoms" => 2),
            "scf" => Dict("max_iterations" => 2, "energy_tolerance" => 1e-6),
        )
        input = joinpath(dir, "input.toml")
        open(input, "w") do io
            TOML.print(io, settings)
        end
        firstdir, seconddir = joinpath(dir, "first"), joinpath(dir, "second")
        first_run = QuanticsKS.run_config(input; output = firstdir)
        pivotfile = joinpath(firstdir, "cross-state.jls")
        @test isfile(pivotfile)
        original = read(pivotfile)
        second_run = QuanticsKS.run_config(
            input;
            output = seconddir,
            checkpoint = joinpath(firstdir, "checkpoint.jls"),
        )
        @test first(second_run.history).scf == 1
        @test length(second_run.history) == 2
        @test !first(second_run.history).has_previous_energy
        @test read(pivotfile) == original
        rows = JSON3.read.(readlines(joinpath(seconddir, "tci_calls.jsonl")))
        @test any(row -> get(row, :pivot_reused, false), rows)
        @test all(x -> all(g -> all(isfinite, g), x.c), second_run.state.orbitals)
        @test QuanticsKS.SCFOptions().energy_tolerance == 1e-6
    end
end
