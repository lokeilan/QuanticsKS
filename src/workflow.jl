# Run the selected method, save provenance, and restore thread settings.

"Execute a prepared calculation in a new directory and write a common receipt."
function run_calculation(calculation::Calculation, output)
    out = abspath(output)
    ispath(out) && throw(ArgumentError("Output already exists: $out; choose a new directory"))
    started = time_ns()
    receipt = Dict{String,Any}(
        "complete" => false,
        "accepted" => false,
        "input" => calculation.input_path,
        "case_index" => calculation.case_index,
        "started_utc" => string(now(UTC)),
    )
    # Restore the caller's thread settings on both success and failure.
    old_blas = BLAS.get_num_threads()
    old_fft = FFTW.get_num_threads()
    try
        result = solve(calculation, out)
        receipt["complete"] = true
        receipt["accepted"] = accepted(result)
        merge!(receipt, acceptance_details(result))
        return (; output = out, accepted = receipt["accepted"], result)
    catch exception
        receipt["error"] = sprint(showerror, exception)
        rethrow()
    finally
        BLAS.set_num_threads(old_blas)
        FFTW.set_num_threads(old_fft)
        if isdir(out)
            save_run_sources(out, calculation)
            receipt["elapsed_seconds"] = (time_ns() - started) / 1e9
            receipt["finished_utc"] = string(now(UTC))
            write_json_file(joinpath(out, "run.json"), receipt)
        end
    end
end

run_calculation(path::AbstractString, output; case_index = 1) =
    run_calculation(read_input(path; case_index), output)

function solve(calculation::Calculation{DenseFD2}, out)
    mkpath(out)
    c = calculation.settings
    H100Reference.write_toml(joinpath(out, "config.toml"), H100Reference.config_dict(c))
    H100Reference.run_reference(c, out)
end

function solve(calculation::Calculation{LowRank1D}, out)
    QuanticsKS.run_config(calculation.settings, calculation.input_path; output = out)
end

function solve(calculation::Calculation{JointK}, out)
    c = calculation.settings
    get(c["execution"], "mode", "solve") == "solve" ||
        throw(ArgumentError("This input is a plan; use a solve input to start SCF"))
    BLAS.set_num_threads(1)
    FFTW.set_num_threads(1)
    layout = periodic_layout(c)
    physics = SiliconJointStep2.build_physics(layout, c["model"]["pseudopotential"])
    load_reference = if haskey(c, "reference")
        () -> begin
            reference = SiliconJointStep2.native_reference(
                c["reference"]["directory"],
                layout;
                storage = Symbol(get(c["execution"], "reference_storage", "mmap")),
            )
            read(c["model"]["pseudopotential"], String) ==
            reference.metadata["model_settings"]["pseudopotential_contents"] || throw(
                ArgumentError("Reference pseudopotential differs from the requested model"),
            )
            reference
        end
    else
        nothing
    end
    # Load dense references after cold-start SCF so they cannot enter the initial orbitals.
    cold = SiliconJointStep2.initialization_mode(c) == "cold"
    reference = cold || load_reference === nothing ? nothing : load_reference()
    SiliconJointStep2.joint_scf(
        physics,
        c,
        reference,
        out;
        case_index = calculation.case_index,
        comparison_reference_loader = cold ? load_reference : nothing,
    )
end

function solve(calculation::Calculation{DenseK}, out)
    c = calculation.settings
    BLAS.set_num_threads(1)
    FFTW.set_num_threads(1)
    physics = SiliconJointStep2.build_physics(periodic_layout(c), c["model"]["pseudopotential"])
    options = get(c, "dense", Dict())
    result = SiliconJointStep2.dense_scf(physics; (Symbol(k) => v for (k, v) in options)...)
    mkpath(out)
    serialize(joinpath(out, "state.jls"), result)
    # The first iteration has no preceding energy. JSON records that as null.
    rows = [
        merge(row, (; energy_change = isfinite(row.energy_change) ? row.energy_change : nothing)) for row in result.history
    ]
    write_json_file(joinpath(out, "scf.json"), rows)
    result
end

# Completion and acceptance are separate; each method keeps its existing acceptance rule.
accepted(result::AbstractDict) = result["accepted_reference"]
accepted(result::QuanticsKS.SCFResult) = result.converged
accepted(result::NamedTuple) =
    hasproperty(result, :receipt) ? result.receipt["accepted"] : result.accepted

acceptance_details(result::AbstractDict) =
    Dict("acceptance_scope" => "dense reference physical criteria")
acceptance_details(result::QuanticsKS.SCFResult) = Dict(
    "acceptance_scope" => string(result.stop_reason),
    "independent_audit_passed" => result.audit.passed,
)
acceptance_details(result::NamedTuple) =
    hasproperty(result, :receipt) ?
    Dict(
        "acceptance_scope" => "joint physical criteria and any configured dense comparison",
        "dense_comparison_performed" => result.receipt["dense_comparison_performed"],
    ) : Dict("acceptance_scope" => "small-grid dense SCF criteria")

solve(::Calculation{PlannedJoint}, out) =
    throw(ArgumentError("This historical proposal is plan-only; choose a solve input"))
