#!/usr/bin/env julia
# CLI: plan checks input; solve writes results to a new directory.
include(joinpath(@__DIR__, "src", "TCITDKS.jl"))
using .TCITDKS
using TOML

function main(args)
    if isempty(args) || args == ["--help"]
        println("Usage: julia --project=. run.jl plan INPUT [CASE]")
        println("       julia --project=. run.jl solve INPUT NEW_OUTPUT [CASE]")
        return 0
    end
    command = args[1]
    if command == "plan"
        length(args) in (2, 3) || error("plan needs INPUT and optional CASE")
        calculation = read_input(args[2]; case_index = length(args) == 3 ? parse(Int, args[3]) : 1)
        TOML.print(stdout, calculation_plan(calculation); sorted = true)
        return 0
    elseif command == "solve"
        length(args) in (3, 4) || error("solve needs INPUT, NEW_OUTPUT and optional CASE")
        result = run_calculation(
            args[2],
            args[3];
            case_index = length(args) == 4 ? parse(Int, args[4]) : 1,
        )
        println("Output: ", result.output)
        println("Accepted by the configured criteria: ", result.accepted)
        return result.accepted ? 0 : 2
    end
    error("Unknown command: $command; use plan or solve")
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
