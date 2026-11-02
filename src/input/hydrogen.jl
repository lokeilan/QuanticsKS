# Validate 1D input and estimate storage before constructing fields or orbitals.

function read_1d_input(system, method, settings, path, case_index)
    case_index == 1 || throw(ArgumentError("1D inputs contain one calculation"))
    if method == "dense_fd2"
        system == "hydrogen_chain" ||
            throw(ArgumentError("dense_fd2 currently accepts a hydrogen chain"))
        configuration = H100Reference.Config(; (Symbol(k) => v for (k, v) in settings)...)
        H100Reference.validate(configuration)
        return Calculation(DenseFD2(), configuration, path, case_index)
    elseif method == "qtt_1d"
        checked_keys(
            settings,
            [
                "run",
                "grid",
                "system",
                "accuracy",
                "scf",
                "tci",
                "guess",
                "refinement",
                "audit",
                "physics",
                "time",
                "drive",
            ],
            "input",
        )
        get(get(settings, "system", Dict()), "kind", "hydrogen_chain") == system ||
            throw(ArgumentError("calculation.system and system.kind must agree"))
        # The same parser is used by the numerical library at execution time.
        QuanticsKS.problem_from_config(settings)
        for (section, kind) in (
            ("accuracy", QuanticsKS.Accuracy),
            ("scf", QuanticsKS.SCFOptions),
            ("tci", QuanticsKS.TCIOptions),
            ("guess", QuanticsKS.InitialGuess),
            ("physics", QuanticsKS.PhysicsNumerics),
            ("refinement", QuanticsKS.RefinementOptions),
        )
            QuanticsKS.validate(QuanticsKS.options_from_dict(kind, get(settings, section, Dict())))
        end
        QuanticsKS.options_from_dict(QuanticsKS.AuditOptions, get(settings, "audit", Dict()))
        run = get(settings, "run", Dict())
        checked_keys(run, ["kind", "output", "checkpoint", "blas_threads"], "run")
        get(run, "kind", "scf") == "scf" ||
            throw(ArgumentError("The common entry point runs ground-state SCF"))
        get(run, "blas_threads", 1) >= 1 || throw(ArgumentError("blas_threads must be positive"))
        return Calculation(LowRank1D(), settings, path, case_index)
    end
    throw(ArgumentError("Unknown 1D method: $method"))
end

function calculation_plan(calculation::Calculation{DenseFD2})
    c = calculation.settings
    merge(
        H100Reference.memory_plan(c),
        Dict(
            "method" => "dense_fd2",
            "model" => H100Reference.model_description(c),
            "required_julia_threads" => c.julia_threads,
            "calculation_started" => false,
        ),
    )
end

function calculation_plan(calculation::Calculation{LowRank1D})
    p = QuanticsKS.problem_from_config(calculation.settings)
    Dict(
        "method" => "qtt_1d",
        "grid_points" => 2^p.grid.R,
        "occupied_orbitals" => length(p.occupations),
        "electrons" => sum(p.occupations),
        "dense_orbital_bytes" => 16 * 2^p.grid.R * length(p.occupations),
        "calculation_started" => false,
        "note" => "Dense storage comparison only; this is not a peak-memory estimate.",
    )
end
