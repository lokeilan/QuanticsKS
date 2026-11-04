using Test, QuanticsKS, TOML, JSON3, LinearAlgebra
const K = QuanticsKS
const ROOT = normpath(joinpath(@__DIR__, ".."))

@testset "One-dimensional input and preflight" begin
    for name in ("h4.toml", "h100-r16.toml", "propagation.toml")
        plan = configuration_plan(joinpath(ROOT, "inputs", name))
        @test !plan["calculation_started"]
        @test plan["grid_points"] > 0
        @test plan["electrons"] > 0
    end
    mktempdir() do dir
        path = joinpath(dir, "input.toml")
        settings = TOML.parsefile(joinpath(ROOT, "inputs", "h4.toml"))
        for change in (
            d -> (d["accuracy"]["orbital_cap"] = -1),
            d -> (d["scf"]["parallel_tasks"] = -1),
            d -> (d["calculation"] = Dict("method" => "dense_fd2")),
            d -> (d["system"]["kind"] = "silicon"),
            d -> (d["run"]["blas_threads"] = 0),
            d -> (d["audit"] = Dict("field_tolerance" => -1.0)),
        )
            bad = deepcopy(settings)
            change(bad)
            open(io -> TOML.print(io, bad), path, "w")
            @test_throws ArgumentError configuration_plan(path)
        end
        @test readdir(dir) == ["input.toml"]
        errors = IOBuffer()
        @test K.main(["unknown"]; err = errors) == 1
        @test occursin("Unknown command", String(take!(errors)))
    end
end

@testset "SCF output, restart, and executable source snapshot" begin
    mktempdir() do dir
        settings = TOML.parsefile(joinpath(ROOT, "inputs", "h4.toml"))
        settings["grid"]["R"] = 5
        settings["scf"]["max_iterations"] = 1
        input = joinpath(dir, "input.toml")
        open(io -> TOML.print(io, settings), input, "w")
        out = joinpath(dir, "scf")
        previous_threads = BLAS.get_num_threads()
        messages = IOBuffer()
        @test K.main(["solve", input, out]; io = messages) == 2
        @test BLAS.get_num_threads() == previous_threads
        text = String(take!(messages))
        @test occursin("Energy/iteration stopping reached: false", text)
        @test occursin("Independent physical audit passed:", text)
        state = load_checkpoint(joinpath(out, "checkpoint.jls"))
        @test length(state.orbitals) == 2
        original = read(joinpath(out, "checkpoint.jls"))
        @test_throws ArgumentError run_config(input; output = out)
        @test read(joinpath(out, "checkpoint.jls")) == original
        @test isfile(joinpath(out, "source", "src", "QuanticsKS.jl"))
        @test isfile(joinpath(out, "source", "Project.toml"))
        @test isfile(joinpath(out, "source", "run.jl"))
        source = joinpath(out, "source")
        plan_text = read(
            `$(Base.julia_cmd()) --startup-file=no --project=$source $(joinpath(source, "run.jl")) plan $input`,
            String,
        )
        @test !TOML.parse(plan_text)["calculation_started"]
        @test JSON3.read(read(joinpath(out, "status.json"), String)).iterations == 1

        resumed = deepcopy(settings)
        resumed["run"]["checkpoint"] = "scf/checkpoint.jls"
        restart_input = joinpath(dir, "restart.toml")
        open(io -> TOML.print(io, resumed), restart_input, "w")
        restarted = cd(mktempdir()) do
            run_config(restart_input; output = joinpath(dir, "restarted"))
        end
        @test first(restarted.history).scf == 1
        @test !first(restarted.history).has_previous_energy
    end
end
