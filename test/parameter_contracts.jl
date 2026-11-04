module ParameterContracts

using QuanticsKS, Test, TOML, JSON3, Random
const K = QuanticsKS
@testset "Exposed parameter contracts" begin
    d = TOML.parsefile(joinpath(@__DIR__, "fixtures", "defaults.toml"))
    for (name, T) in (
        ("accuracy", Accuracy),
        ("scf", SCFOptions),
        ("tci", TCIOptions),
        ("guess", InitialGuess),
        ("physics", PhysicsNumerics),
        ("refinement", RefinementOptions),
        ("audit", AuditOptions),
        ("time", TimeOptions),
    )
        @test K.option_dict(K.options_from_dict(T, d[name])) == K.option_dict(T())
    end
    @test_throws ArgumentError LocalPotential1D(Inf)
    @test_throws ArgumentError Problem(
        PeriodicGrid1D(5, 4),
        SoftCoulombLSDA1D([0.0, 2.0]; softening = 2, xc = :none),
        [2.0],
    )
    @test_throws ArgumentError K.validate(InitialGuess(envelope_amplitude = NaN))
    p = Problem(PeriodicGrid1D(14, 1.0), LocalPotential1D(0.0), [1.0])
    j = 10037
    localized = state_from_functions(
        p,
        [x -> x == j / 2^14 ? 1.0 : 0.0];
        tci = TCIOptions(initial_pivots = [j], initial_random_pivots = 0),
    )
    @test abs(K.Q.Evaluator(first(localized.orbitals))(j)) ≈ 1 atol = 1e-12
    @test abs(K.Q.Evaluator(first(localized.orbitals))(j + 1)) < 1e-12
    @test_throws ArgumentError state_from_functions(
        p,
        [x -> 1.0];
        tci = TCIOptions(initial_pivots = [2^14]),
    )
    mktempdir() do dir
        options = TCIOptions(
            max_iterations = 7,
            check_history = 1,
            global_pivot_search = 2,
            max_global_pivots = 2,
            initial_random_pivots = 3,
            post_rounding_factor = 0.02,
            scalar_cache_capacity = 31,
            orbital_cache_capacity = 17,
            environment_cache_bits = 2,
        )
        rec = K.recorder(dir, options)
        try
            Random.seed!(12)
            K.Q.cross(
                rec,
                5,
                j -> exp(0.2cos(2pi * j / 32));
                tol = 1e-8,
                cap = 4,
                label = "parameter_probe",
            )
        finally
            close(rec)
        end
        row = JSON3.read(readlines(joinpath(dir, "tci_calls.jsonl"))[1])
        @test row.cache_mode == "dense_addressed"
        @test row.cache_peak_cap == 32
        @test row.rank_cap == 4
        @test row.post_rounding_tolerance ≈ 2e-10
        @test row.post_rounding_seconds >= 0.0
        @test row.construction == "tci_prrlu_with_svd_rounding"
    end
end

end
