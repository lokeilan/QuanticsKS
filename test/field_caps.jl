module FieldCapChecks
using QuanticsKS, Test, LinearAlgebra, Random, JSON3, TOML
const K = QuanticsKS
const Q = K.Q
BLAS.set_num_threads(1)

@testset "Independent field budgets and legacy fallback" begin
    p = hydrogen_chain(R = 6, natoms = 2)
    legacy = Accuracy(field_cap = 16)
    @test all(==(16), values(K.field_caps(legacy)))
    a = Accuracy(field_cap = 16, external_cap = 5, kernel_cap = 6,
                 density_cap = 7, hartree_cap = 3, effective_potential_cap = 4,
                 xc_energy_cap = 8, drive_cap = 2)
    @test K.validate(a) === a
    for name in keys(K.field_caps(a))
        @test_throws ArgumentError K.validate(Accuracy(; name => 0))
    end
    partial = Accuracy(field_cap = 13, kernel_cap = 2)
    @test K.field_caps(partial).density_cap == 13
    @test K.field_caps(partial).kernel_cap == 2
    cfg = K.solver_config(p, a)
    @test cfg.opcap == 4 && cfg.density_cap == 7
    physics = PhysicsNumerics(xc_table_points = 32)
    Random.seed!(20261106)
    rec = K.recorder(nothing, TCIOptions())
    state = K.initial_state(p, legacy, rec, InitialGuess())
    close(rec)
    mktempdir() do dir
        rec = K.recorder(dir, TCIOptions())
        try
            ops = K.build_operators(p, a, rec, physics)
            rho = K.density(p, state.orbitals, a, rec; renormalize = true)
            H = K.effective_hamiltonian(p, rho, ops, a, rec)
            @test Q.rank(ops.external) <= 5
            @test maximum(w -> size(w, 4), ops.interaction) <= 2 * 6
            @test Q.rank(rho) <= 7
            @test Q.rank(K.hartree(rho, p, ops, a)) <= 3
            @test Q.rank(H.potential) <= 4
            # A generic density requires more bonds than the potential budget.
            random_density = Q.cross(rec, 6, j -> 1 + sin((j + 1)^2); tol = 1e-12,
                                     cap = 8, label = "mixing_input")
            mixed = Q.materialize_density(Q.linear_expression((random_density,), (1.0,)), cfg, rec)
            @test 4 < Q.rank(mixed) <= 7
            @test K.spacing(p.grid) * real(Q.inner(Q.constant_tt(6), mixed)) ≈ 2
            drive = FunctionDrive((x, t) -> 0.2cos(2pi * x / p.grid.L), "field-cap-test")
            energy = K.total_energy(p, state.orbitals, rho, ops, a, rec; time = 0.5, drive)
            @test isfinite(energy.total)
            potential = [real(Q.Evaluator(H.potential)(j)) for j in 0:63]
            projected = K.spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = 1e-10)
            @test Q.rank(projected.potential) <= 4
            # Audit scaling applies to effective component caps, including explicit overrides.
            audited = K.audit_accuracy(a, AuditOptions(rank_multiplier = 3))
            @test all(K.field_caps(audited)[name] == 3cap for (name, cap) in pairs(K.field_caps(a)))
            @test all(==(48), values(K.field_caps(K.audit_accuracy(legacy, AuditOptions(rank_multiplier = 3)))))
            rec.scf = 100
            check = K.audit_state(state, rho, a, AuditOptions(rank_multiplier = 3), physics, rec;
                                  reference_energy = energy.total)
            @test isfinite(check.max_residual)
        finally
            close(rec)
        end
        rows = JSON3.read.(readlines(joinpath(dir, "tci_calls.jsonl")))
        expected = Dict("external" => 5, "interaction" => 6, "density" => 7,
                        "effective_potential" => 4, "xc_energy_density" => 8,
                        "drive_energy" => 2, "fixed_projection_potential" => 4)
        for (label, cap) in expected
            selected = filter(r -> r.label == label && r.scf == 0, rows)
            @test !isempty(selected)
            @test all(r -> r.rank_cap == cap, selected)
        end
        for (label, cap) in (("external", 15), ("interaction", 18), ("audit_density", 21),
                             ("effective_potential", 12), ("xc_energy_density", 24))
            selected = filter(r -> r.label == label && r.scf == 100, rows)
            @test !isempty(selected)
            @test all(r -> r.rank_cap == cap, selected)
        end
    end

    # Local-only H reuses the external cores unless its consumer requests a lower cap.
    local_problem = Problem(PeriodicGrid1D(6, 8),
                           LocalPotential1D(x -> exp(0.7cos(2pi * x / 8)); id = "field-cap-local"), [1.0])
    rec = K.recorder(nothing, TCIOptions())
    try
        ops = K.build_operators(local_problem, legacy, rec, physics)
        rho = Q.constant_tt(6)
        H = K.effective_hamiltonian(local_problem, rho, ops, legacy, rec)
        @test H.potential === ops.external
        calls = rec.calls
        smaller = Accuracy(field_cap = 16, effective_potential_cap = 2)
        H = K.effective_hamiltonian(local_problem, rho, ops, smaller, rec)
        @test Q.rank(H.potential) <= 2
        @test rec.calls == calls + 1
        @test Q.rank(ops.external) > 2
    finally
        close(rec)
    end
end

@testset "Field cap configuration and fixed-potential construction" begin
    mktempdir() do dir
        input = TOML.parsefile(joinpath(@__DIR__, "..", "inputs", "h4.toml"))
        input["accuracy"]["external_cap"] = 6
        input["accuracy"]["effective_potential_cap"] = 4
        path = joinpath(dir, "input.toml")
        open(io -> TOML.print(io, input), path, "w")
        caps = configuration_plan(path)["resolved_field_caps"]
        @test caps["external_cap"] == 6 && caps["density_cap"] == 32
        @test caps["effective_potential_cap"] == 4
        p = Problem(PeriodicGrid1D(5, 8), LocalPotential1D(0.0), [1.0])
        a = Accuracy(field_cap = 16, external_cap = 8, effective_potential_cap = 3)
        state = state_from_functions(p, [x -> 1.0]; accuracy = a)
        v = [exp(0.7cos(2pi * j / 32)) for j in 0:31]
        output = joinpath(dir, "spectrum")
        fixed_spectrum(state, v; accuracy = a, output,
                       spectrum = FixedSpectrumOptions(states = 1, occupied_states = 1,
                                                       virtual_wavevectors = Int[],
                                                       min_iterations = 2, max_iterations = 2))
        config = JSON3.read(read(joinpath(output, "config.json"), String))
        @test config.resolved_field_caps.effective_potential_cap == 3
        rows = JSON3.read.(readlines(joinpath(output, "tci_calls.jsonl")))
        @test only(filter(r -> r.label == "fixed_potential", rows)).rank_cap == 3
    end
end
end
