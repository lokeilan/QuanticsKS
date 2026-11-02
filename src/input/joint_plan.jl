# Inspect legacy joint plans without allocating orbitals or starting SCF.

"Read and audit a proposal. This function never starts SCF or reads old checkpoints."
function read_plan(path)
    c = TOML.parsefile(path)
    c["representation"]["kind"] == "joint_xk" || error("joint_xk required")
    c["representation"]["order"] == "space_k_axis_msb" || error("explicit site order required")
    c["representation"]["samples"] == "physical_cell_periodic_orbitals" &&
        c["representation"]["k_weights"] == "uniform_normalized" ||
        error("physical samples and uniform normalized mesh required")
    m = c["model"]
    m["cell"] == "conventional" && m["occupied_bands"] == 16 && m["electrons"] == 32 ||
        error("eight-atom, sixteen-band conventional silicon required")
    m["functional"] == "PZ-LDA" && m["kinetic"] == "bloch_phase_fd" && m["fd_order"] == 8 ||
        error("physical model identity mismatch")
    l = JointLayout(m["space_bits"], m["k_bits"]; a_bohr = m["a_bohr"], k_shift = m["k_shift"])
    pseudo = normpath(joinpath(dirname(abspath(path)), m["pseudopotential"]))
    isfile(pseudo) || error("missing pseudopotential: $pseudo")
    head = parse.(Float64, split(first(readlines(pseudo))))
    length(head) == 4 && head[3] == 2 && head[4] == 4 ||
        error("four-valence s/p/d KY asset required")
    c["execution"]["mode"] == "plan" || error("only plan mode is implemented")
    !c["execution"]["launch_production"] || error("production launch is not implemented")
    !c["patching"]["enabled"] || error("adaptive patching is a proposal, not implemented")
    c["inner"]["method"] == "block_green" &&
        c["inner"]["stop"] == "occupied_eigenvalue_sum_change" ||
        error("archive-derived block Green inner scheme required")
    c["scf"]["stop"] == "dft_total_energy_change" || error("outer DFT energy stop required")
    c["inner"]["occupied_states"] == 16 && 16 <= c["inner"]["states"] <= prod(shape(l)) ||
        error("invalid fixed-H state count")
    for section in ("inner", "scf")
        o = c[section]
        isfinite(o["energy_tolerance_Ha"]) &&
            o["energy_tolerance_Ha"] > 0 &&
            2 <= o["minimum_iterations"] <= o["maximum_iterations"] ||
            error("invalid energy stopping controls")
    end
    0 < c["scf"]["mixing"] <= 1 && c["scf"]["mixing_method"] == "linear" || error("invalid mixing")
    isfinite(c["inner"]["green_shift"]) && c["inner"]["green_shift"] > 0 ||
        error("positive finite Green shift required")
    c["validation"]["fresh_hamiltonian"] &&
        c["validation"]["per_k_constraints"] &&
        !c["validation"]["renormalize_density"] || error("raw per-k fresh validation required")
    all(t -> isfinite(t) && 0 < t < 1, c["interpolation"]["tolerances"]) ||
        error("invalid TCI tolerances")
    caps = [c["accuracy"][k] for k in ("orbital_cap", "work_cap", "field_cap")]
    all(>(0), caps) && caps[2] >= caps[1] ||
        error("positive caps; work cap >= orbital cap required")
    dense = 16 * m["occupied_bands"] * prod(shape(l)) * prod(kshape(l))
    rank_bound = [min(caps[1], 1 << i, 1 << (sites(l) - i)) for i = 0:sites(l)]
    core_bound =
        16 * m["occupied_bands"] * sum(2rank_bound[i] * rank_bound[i + 1] for i = 1:sites(l))
    (;
        status = "plan_only",
        configuration = c,
        layout = l,
        pseudopotential = pseudo,
        spatial_shape = shape(l),
        k_shape = kshape(l),
        joint_sites = sites(l),
        dense_orbital_bytes = dense,
        cap_bound_core_bytes = core_bound,
        production_solver_implemented = false,
        adaptive_patching_implemented = false,
    )
end
