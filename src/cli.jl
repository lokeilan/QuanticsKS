# Keep command-line handling separate from configuration and numerical work.
const CLI_USAGE = """
Usage:
  julia --project=. run.jl plan INPUT.toml
  julia --project=. run.jl solve INPUT.toml NEW_OUTPUT
"""

function main(args = ARGS; io = stdout, err = stderr)
    if isempty(args) || args == ["--help"]
        print(io, CLI_USAGE)
        return 0
    end
    try
        if args[1] == "plan"
            length(args) == 2 || throw(ArgumentError("plan needs INPUT.toml"))
            TOML.print(io, configuration_plan(args[2]); sorted = true)
            return 0
        elseif args[1] == "solve"
            length(args) == 3 || throw(ArgumentError("solve needs INPUT.toml and NEW_OUTPUT"))
            result = run_config(args[2]; output = args[3])
            println(io, "Output: ", abspath(args[3]))
            if result isa SCFResult
                println(io, "Energy/iteration stopping reached: ", result.converged)
                println(io, "Independent physical audit passed: ", result.audit.passed)
                return result.converged ? 0 : 2
            end
            println(io, "Propagation steps completed: ", result.completed_steps)
            return 0
        end
        throw(ArgumentError("Unknown command: $(args[1]); use plan or solve"))
    catch exception
        exception isa InterruptException && rethrow()
        println(err, sprint(showerror, exception))
        return 1
    end
end
