# Resolve the system and method here, including legacy input formats.

abstract type NumericalMethod end
struct DenseFD2 <: NumericalMethod end
struct LowRank1D <: NumericalMethod end
struct JointK <: NumericalMethod end
struct DenseK <: NumericalMethod end
struct PlannedJoint <: NumericalMethod end

"Input after validation; paths are resolved relative to the input file."
struct Calculation{M<:NumericalMethod,C}
    method::M
    settings::C
    input_path::String
    case_index::Int
end

function checked_keys(settings, allowed, section)
    unknown = setdiff(collect(keys(settings)), allowed)
    isempty(unknown) || throw(ArgumentError("Unknown $section options: $(join(unknown, ", "))"))
end

function input_path(input_file, value)
    isabspath(value) ? normpath(value) : normpath(joinpath(dirname(input_file), value))
end

"Read a new or legacy input without constructing fields or starting SCF."
function read_input(path; case_index = 1)
    path = abspath(path)
    settings = TOML.parsefile(path)
    selection = pop!(settings, "calculation", Dict{String,Any}())
    checked_keys(selection, ["system", "method"], "calculation")
    case_index isa Integer && case_index >= 1 || throw(ArgumentError("case_index must be positive"))

    # Legacy inputs are accepted at this one location. The solver never has to
    # distinguish a historical file from a new one.
    system = get(
        selection,
        "system",
        haskey(settings, "model") ? "silicon" :
        get(get(settings, "system", Dict()), "kind", "hydrogen_chain"),
    )
    if system == "hydrogen_chain" || system == "soft_coulomb" || system == "cosine_potential"
        default_method = haskey(settings, "system") ? "qtt_1d" : "dense_fd2"
        method = get(selection, "method", default_method)
        return read_1d_input(system, method, settings, path, case_index)
    elseif system == "silicon"
        method = get(selection, "method", "joint_xk")
        return read_periodic_input(method, settings, path, case_index)
    end
    throw(ArgumentError("Unsupported system: $system"))
end
