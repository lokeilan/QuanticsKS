# Validate silicon input and resolve paths; spatial and k grids are separate.

function read_periodic_input(method, settings, path, case_index)
    method in ("joint_xk", "dense_k") || throw(ArgumentError("Unknown periodic method: $method"))
    checked_keys(
        settings,
        [
            "model",
            "reference",
            "interpolation",
            "accuracy",
            "inner",
            "scf",
            "validation",
            "execution",
            "representation",
            "patching",
            "dense",
        ],
        "input",
    )
    model = settings["model"]
    checked_keys(
        model,
        [
            "cell",
            "a_bohr",
            "electrons",
            "occupied_bands",
            "space_bits",
            "k_bits",
            "k_shift",
            "pseudopotential",
            "functional",
            "kinetic",
            "fd_order",
        ],
        "model",
    )
    model["cell"] == "conventional" &&
        model["electrons"] == 32 &&
        model["occupied_bands"] == 16 &&
        model["functional"] == "PZ-LDA" &&
        model["kinetic"] == "bloch_phase_fd" &&
        model["fd_order"] == 8 || throw(
        ArgumentError(
            "Expected conventional 8-atom silicon, 16 occupied bands, PZ-LDA and Bloch FD8",
        ),
    )
    layout = periodic_layout(settings)
    model["pseudopotential"] = input_path(path, model["pseudopotential"])
    isfile(model["pseudopotential"]) ||
        throw(ArgumentError("Missing pseudopotential: $(model["pseudopotential"])"))
    execution = get!(settings, "execution", Dict{String,Any}())
    if haskey(execution, "initial_checkpoint")
        execution["initial_checkpoint"] = input_path(path, execution["initial_checkpoint"])
    end
    if haskey(settings, "reference")
        reference = settings["reference"]
        reference["directory"] = input_path(path, reference["directory"])
    end
    get(get(settings, "patching", Dict()), "enabled", false) &&
        throw(ArgumentError("Adaptive patching is not implemented"))
    if method == "joint_xk" && get(execution, "mode", "solve") == "plan"
        SiliconJointStep2.read_plan(path)
        case_index <= length(settings["interpolation"]["tolerances"]) ||
            throw(ArgumentError("case_index exceeds the tolerance list"))
        return Calculation(PlannedJoint(), settings, path, case_index)
    end
    if method == "dense_k"
        case_index == 1 || throw(ArgumentError("dense_k has one case"))
        dense = get(settings, "dense", Dict())
        checked_keys(
            dense,
            [
                "max_steps",
                "mixing",
                "density_tolerance",
                "residual_tolerance",
                "energy_tolerance",
                "maximum_points",
            ],
            "dense",
        )
        prod(SiliconJointStep2.shape(layout)) <= get(dense, "maximum_points", 512) ||
            throw(ArgumentError("Dense matrix exceeds the declared point budget"))
        return Calculation(DenseK(), settings, path, case_index)
    end
    validate_joint_controls(settings, layout, case_index)
    Calculation(JointK(), settings, path, case_index)
end

function periodic_layout(settings)
    m = settings["model"]
    SiliconJointStep2.JointLayout(
        m["space_bits"],
        m["k_bits"];
        a_bohr = m["a_bohr"],
        k_shift = m["k_shift"],
    )
end

function validate_joint_controls(c, layout, case_index)
    tolerances = c["interpolation"]["tolerances"]
    1 <= case_index <= length(tolerances) ||
        throw(ArgumentError("case_index exceeds the tolerance list"))
    all(t -> isfinite(t) && 0 < t < 1, tolerances) ||
        throw(ArgumentError("Invalid interpolation tolerance"))
    for section in ("scf", "inner")
        options = c[section]
        1 <= options["minimum_iterations"] <= options["maximum_iterations"] ||
            throw(ArgumentError("Invalid $section iteration limits"))
        isfinite(options["energy_tolerance_Ha"]) && options["energy_tolerance_Ha"] > 0 ||
            throw(ArgumentError("Invalid $section energy tolerance"))
    end
    0 < c["scf"]["mixing"] <= 1 || throw(ArgumentError("Mixing must be in (0, 1]"))
    c["inner"]["method"] in ("block_green", "block_residual") ||
        throw(ArgumentError("Unknown inner solver"))
    c["inner"]["states"] == c["inner"]["occupied_states"] == 16 ||
        throw(ArgumentError("The joint solver retains sixteen occupied states"))
    caps = [c["accuracy"][name] for name in ("orbital_cap", "work_cap", "field_cap")]
    all(>(0), caps) && caps[2] >= caps[1] || throw(ArgumentError("Invalid bond caps"))
    kind = get(get(c, "representation", Dict()), "kind", "joint_xk")
    kind in ("joint_xk", "fcc_joint_xk_band") ||
        throw(ArgumentError("Unknown joint representation"))
    if kind == "fcc_joint_xk_band"
        layout.space_bits[1] >= 2 && layout.space_bits[2] >= 2 ||
            throw(ArgumentError("FCC representation needs at least two x and y bits"))
        c["representation"]["blocks"] == c["representation"]["bands_per_block"] == 4 ||
            throw(ArgumentError("FCC representation uses four blocks of four bands"))
    end
    mode = SiliconJointStep2.initialization_mode(c)
    mode in ("cold", "dense_perturbed", "checkpoint") ||
        throw(ArgumentError("Unknown initialization"))
    mode == "dense_perturbed" &&
        !haskey(c, "reference") &&
        throw(ArgumentError("dense_perturbed requires a reference directory"))
    checkpoint = get(c["execution"], "initial_checkpoint", "")
    mode == "checkpoint" ?
    (isfile(checkpoint) || throw(ArgumentError("Missing initial checkpoint"))) :
    (
        isempty(checkpoint) ||
        throw(ArgumentError("initial_checkpoint requires checkpoint initialization"))
    )
end

function calculation_plan(calculation::Calculation{M}) where {M<:Union{JointK,DenseK,PlannedJoint}}
    c = calculation.settings
    layout = periodic_layout(c)
    Dict(
        "method" => (M == DenseK ? "dense_k" : "joint_xk"),
        "representation" => get(get(c, "representation", Dict()), "kind", "dense"),
        "spatial_shape" => collect(SiliconJointStep2.shape(layout)),
        "k_shape" => collect(SiliconJointStep2.kshape(layout)),
        "occupied_orbitals_per_k" => 16,
        "dense_orbital_bytes" =>
            16 *
            16 *
            prod(SiliconJointStep2.shape(layout)) *
            prod(SiliconJointStep2.kshape(layout)),
        "case_index" => calculation.case_index,
        "calculation_started" => false,
        "execution_allowed" => M != PlannedJoint,
        "note" => "Dense storage comparison only; fields, orbitals and reference data have not been loaded.",
    )
end
