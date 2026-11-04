module StateWorkflowChecks
using Test, QuanticsKS, LinearAlgebra, Random, Serialization, TOML
const K = QuanticsKS
const Q = K.Q
BLAS.set_num_threads(1)
Random.seed!(310)
const tight = Accuracy(
    field_tolerance = 1e-12,
    work_tolerance = 1e-10,
    storage_tolerance = 1e-12,
    algebra_tolerance = 1e-14,
    density_tolerance = 1e-12,
    kernel_tolerance = 1e-13,
    external_tolerance = 1e-13,
    xc_energy_tolerance = 1e-12,
    orthogonalization_tolerance = 1e-13,
    green_rhs_tolerance = 1e-12,
    orbital_cap = 32,
    field_cap = 32,
    work_cap = 64,
)
dense(s) = hcat(
    [
        [Q.Evaluator(x; cachebits = 0)(j) for j = 0:(length(s.problem.grid) - 1)] for
        x in s.orbitals
    ]...,
)
function fd2(p)
    N = length(p.grid)
    h = K.spacing(p.grid)
    T = Matrix{ComplexF64}(I, N, N) / (p.mass * h^2)
    for j = 1:N
        T[j, mod1(j + 1, N)] = -1 / (2p.mass * h^2)
        T[j, mod1(j - 1, N)] = -1 / (2p.mass * h^2)
    end
    T
end

@testset "Configuration contracts" begin
    @test_throws ArgumentError PeriodicGrid1D(1, 8)
    @test_throws ArgumentError SoftCoulombLSDA1D([0.0, 4.0]; softening = 2)
    @test SoftCoulombLSDA1D([0.0, 4.0]; softening = 2, xc = :none).softening == 2
    @test_throws ArgumentError Problem(PeriodicGrid1D(5, 8), SoftCoulombLSDA1D([0.0, 4.0]), [1.0])
    @test_throws ArgumentError K.options_from_dict(Accuracy, Dict("orbital_cpa" => 32))
    @test_throws ArgumentError K.validate(TCIOptions(environment_cache_bits = 30))
    @test_throws ArgumentError K.validate(SCFOptions(mixing = 0.0))
    @test_throws ArgumentError K.validate(TimeOptions(dt = -0.1))
    @test_throws ArgumentError K.validate(RefinementOptions(degree = 4))
    p = K.problem_from_config(TOML.parsefile(joinpath(@__DIR__, "..", "inputs", "h100-r16.toml")))
    @test p.grid.R == 16 &&
          p.grid.L == 400 &&
          sum(p.occupations) == 100 &&
          length(p.occupations) == 50
end

@testset "Model, state, checkpoint and refinement" begin
    g = PeriodicGrid1D(5, 8)
    model = LocalPotential1D(x -> 0.0; id = "free")
    p = Problem(g, model, [1.0, 1.0]; mass = 2)
    s = state_from_functions(p, [x -> 1, x -> cis(2pi * x / g.L)]; accuracy = tight)
    o = observe(s; accuracy = tight)
    N = length(g)
    h = K.spacing(g)
    @test o.electron_number ≈ 2 atol = 1e-11
    @test o.gram_defect < 1e-11
    @test o.energy.total ≈ (1 - cos(2pi / N)) / (p.mass * h^2) atol = 1e-10
    @test size(bond_profiles(s)) == (2, 4)
    fine = Problem(PeriodicGrid1D(7, 8), model, p.occupations; mass = 2)
    f = refine_state(s, fine; accuracy = tight, refinement = RefinementOptions(tolerance = 1e-12))
    truth = state_from_functions(fine, [x -> 1, x -> cis(2pi * x / g.L)]; accuracy = tight)
    @test norm(dense(f) - dense(truth)) < 3e-7
    @test_throws ArgumentError refine_state(s, hydrogen_chain(R = 7, natoms = 2))
    mktempdir() do dir
        file = joinpath(dir, "state.jls")
        save_checkpoint(file, s)
        @test_throws ArgumentError load_checkpoint(file)
        restored = load_checkpoint(file; model)
        @test dense(restored) == dense(s)
        @test_throws ArgumentError load_checkpoint(
            file;
            model = LocalPotential1D(x -> 1.0; id = "different"),
        )
        d = deserialize(file)
        d["schema_version"] = 99
        serialize(file, d)
        @test_throws ArgumentError load_checkpoint(file; model)
    end
    # Exercise a physical parameter independent of the nuclear count.
    hartree_model =
        SoftCoulombLSDA1D([0.0, 4.0], [1.0, 1.0]; softening = 1.7, coupling = 0.5, xc = :none)
    @test K.kernel_value(hartree_model, 0.2, 8) ≈ 0.5Q.soft(0.2 / 1.7, 8 / 1.7) / 1.7
    @test_throws ArgumentError Problem(g, SoftCoulombLSDA1D([0.0, 4.0]), [2.0]; mass = 2)
end

@testset "Cayley phase and second-order propagation" begin
    p = Problem(PeriodicGrid1D(5, 8), LocalPotential1D(x -> 0.0; id = "free"), [1.0])
    s = state_from_functions(p, [x -> cis(2pi * x / 8)]; accuracy = tight)
    settings(dt, n) = TimeOptions(
        dt = dt,
        steps = n,
        phase_tolerance = 1e-13,
        kinetic_tolerance = 1e-14,
        output_tolerance = 1e-14,
        phase_cap = 32,
        kinetic_cap = 64,
        output_cap = 32,
        observe_every = n,
        snapshot_every = n,
    )
    result = propagate(s; accuracy = tight, time = settings(0.1, 4))
    e = (1 - cos(2pi / 32)) / K.spacing(p.grid)^2
    factor = ((1 - im * 0.1e / 2) / (1 + im * 0.1e / 2))^4
    @test norm(dense(result.state) - factor * dense(s)) < 2e-10
    @test abs(last(result.observations).electron_number - 1) < 1e-11
    @test result.state.time ≈ 0.4
    # A spatially nonconstant Hamiltonian supplies a true splitting-error test.
    model = LocalPotential1D(x -> 0.3cos(2pi * x / 8); id = "cosine-test")
    p = Problem(p.grid, model, [1.0])
    s = state_from_functions(p, [x -> 1 + 0.2cos(2pi * x / 8)]; accuracy = tight)
    H = fd2(p) + Diagonal([model.potential(j * K.spacing(p.grid)) for j = 0:31])
    exact = exp(-im * 0.4H) * dense(s)
    errors = Float64[]
    for (dt, n) in ((0.1, 4), (0.05, 8), (0.025, 16))
        r = propagate(s; accuracy = tight, time = settings(dt, n))
        push!(errors, norm(dense(r.state) - exact))
        @test abs(last(r.observations).electron_number - 1) < 1e-10
    end
    @test 3.7 < errors[1] / errors[2] < 4.3
    @test 3.7 < errors[2] / errors[3] < 4.3
    println("Time-step errors: ", errors)
    # No hidden normalization: initially non-unit amplitudes keep their charge.
    scaled = OrbitalState(p, [Q.scale(first(s.orbitals), 0.9)])
    r = propagate(scaled; accuracy = tight, time = settings(0.02, 2))
    @test last(r.observations).electron_number ≈ 0.81 atol = 1e-10
    @test_throws ArgumentError propagate(s; time = TimeOptions(dt = 0.0))
end

@testset "SCF, recording and postprocessing" begin
    p = hydrogen_chain(R = 5, natoms = 2)
    mktempdir() do dir
        out = joinpath(dir, "h2")
        result = ground_state(p; scf = SCFOptions(max_iterations = 30), output = out)
        @test result.converged
        @test last(result.history).energy_change < 1e-5
        @test length(result.history) >= 2
        @test result.audit.electron_number ≈ 2 atol = 1e-7
        @test_throws ArgumentError ground_state(p; output = out)
        restored = load_checkpoint(joinpath(out, "checkpoint.jls"))
        @test dense(restored) == dense(result.state)
        postprocess(out; samples = 16)
        @test isfile(joinpath(out, "observables", "bond_profiles.csv"))
        @test isfile(joinpath(out, "observables", "trajectory.json"))
    end
end
end
