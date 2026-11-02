# Save inputs, replay settings, and source hashes; resolve paths from their input files.

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"Freeze executable sources and the original input, with content hashes."
function save_run_sources(out, calculation)
    destination = joinpath(out, "source")
    mkpath(destination)
    files = ["Project.toml", "Manifest.toml", "run.jl"]
    for folder in ("src", "lib/QuanticsKS")
        for (directory, _, names) in walkdir(joinpath(PROJECT_ROOT, folder)), name in names
            endswith(name, ".jl") || name in ("Project.toml", "Manifest.toml") || continue
            push!(files, relpath(joinpath(directory, name), PROJECT_ROOT))
        end
    end
    hashes = Dict{String,String}()
    for file in files
        target = joinpath(destination, file)
        mkpath(dirname(target))
        cp(joinpath(PROJECT_ROOT, file), target)
        hashes[replace(file, '\\' => '/')] = bytes2hex(sha256(read(target)))
    end
    cp(calculation.input_path, joinpath(out, "input.toml"); force = true)
    hashes["input.toml"] = bytes2hex(sha256(read(calculation.input_path)))
    if calculation.settings isa AbstractDict && haskey(calculation.settings, "model")
        pseudo = calculation.settings["model"]["pseudopotential"]
        cp(pseudo, joinpath(out, "pseudopotential.dat"))
        hashes["pseudopotential.dat"] = bytes2hex(sha256(read(pseudo)))
    end
    replay =
        calculation.settings isa AbstractDict ? deepcopy(calculation.settings) :
        H100Reference.config_dict(calculation.settings)
    original = TOML.parsefile(calculation.input_path)
    haskey(original, "calculation") && (replay["calculation"] = original["calculation"])
    haskey(replay, "model") && (replay["model"]["pseudopotential"] = "pseudopotential.dat")
    if haskey(replay, "run") && haskey(replay["run"], "checkpoint")
        replay["run"]["checkpoint"] =
            input_path(calculation.input_path, replay["run"]["checkpoint"])
    end
    open(joinpath(out, "replay.toml"), "w") do io
        TOML.print(io, replay; sorted = true)
    end
    open(joinpath(out, "source-sha256.toml"), "w") do io
        TOML.print(io, hashes; sorted = true)
    end
end

write_json_file(path, data) = open(io -> JSON3.write(io, data), path, "w")
