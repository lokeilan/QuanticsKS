module FixedSpectrumChecks
using QuanticsKS, Test, LinearAlgebra, Random
BLAS.set_num_threads(1)
const K = QuanticsKS
@testset "Fixed H separates solved and occupied states" for (method, basis, nstates, wavevectors) in
                                                            (
    (:block_residual, :projected, 4, [-1, 2]),
    (:block_residual, :direct, 4, [-1, 2]),
    (:block_green, :direct, 5, [-1, 2, -2]),
)
    p = Problem(
        PeriodicGrid1D(5, 8),
        LocalPotential1D(x -> 0.2cos(2pi * x / 8); id = "fixed-test"),
        [2.0, 2.0],
    )
    a = Accuracy(
        field_tolerance = 1e-12,
        work_tolerance = 1e-10,
        storage_tolerance = 1e-12,
        algebra_tolerance = 1e-13,
        orbital_cap = 32,
        field_cap = 32,
        work_cap = 64,
        orthogonalization_tolerance = 1e-12,
        green_rhs_tolerance = 1e-12,
    )
    state = state_from_functions(p, [x -> 1, x -> cis(2pi * x / 8)]; accuracy = a)
    v = [p.model.potential(j * 8 / 32) for j = 0:31]
    # A fixed minimum budget checks all eigenpairs, including unoccupied orbitals.
    o = FixedSpectrumOptions(
        states = nstates,
        occupied_states = 2,
        virtual_wavevectors = wavevectors,
        virtual_envelope_amplitude = 0.1,
        virtual_envelope_period = 8.0,
        min_iterations = 40,
        max_iterations = 60,
        energy_tolerance = 1e-12,
        direction_residual_floor = 1e-10,
        method = method,
        search_basis = basis,
    )
    r = fixed_spectrum(
        state,
        v;
        spectrum = o,
        accuracy = a,
        solver = SCFOptions(green_polish_every = 1),
    )
    @test length(r.orbitals) == nstates
    @test length(r.occupied.orbitals) == 2 && sum(r.occupied.problem.occupations) == 4
    @test last(r.history).max_residual < first(r.history).max_residual / 2
    @test last(r.history).trace < first(r.history).trace
    @test last(r.history).gram_defect < 1e-8
    H = Matrix(Diagonal(16 .+ v))
    for i = 1:32
        H[i, mod1(i + 1, 32)] = -8
        H[i, mod1(i - 1, 32)] = -8
    end
    reference = eigvals(Hermitian(H))[1:nstates]
    @test maximum(abs.(r.eigenvalues - reference)) < 1e-9
    @test r.converged
    @test_throws ArgumentError fixed_spectrum(
        state,
        v;
        spectrum = FixedSpectrumOptions(
            states = 4,
            occupied_states = 3,
            virtual_wavevectors = [-1],
        ),
    )
    if basis == :direct
        mktempdir() do dir
            cp = joinpath(dir, "spectrum.jls")
            hash = bytes2hex(K.sha256(reinterpret(UInt8, v)))
            K.spectrum_checkpoint(dir, p, r.orbitals, r.eigenvalues, r.history, o, hash)
            resumed = fixed_spectrum(state, v; spectrum = o, accuracy = a, resume_from = cp)
            @test resumed.converged && last(resumed.history).iteration >= o.min_iterations
            @test first(resumed.history).eigenvalues ≈ last(r.history).eigenvalues atol = 1e-12
            @test_throws ArgumentError fixed_spectrum(
                state,
                v .+ 1e-6;
                spectrum = o,
                accuracy = a,
                resume_from = cp,
            )
        end
    end
end

@testset "Fixed-H convergence uses energy change only" begin
    p = Problem(
        PeriodicGrid1D(5, 8),
        LocalPotential1D(x -> 0.2cos(2pi * x / 8); id = "energy-test"),
        [1.5, 0.5];
        mass = 0.7,
    )
    state = state_from_functions(p, [x -> 1, x -> cis(2pi * x / 8)])
    v = [p.model.potential(j * 0.25) for j = 0:31]
    o = FixedSpectrumOptions(
        states = 5,
        occupied_states = 2,
        virtual_wavevectors = [-1, 2, -2],
        method = :block_green,
        energy_tolerance = 1.0,
        min_iterations = 2,
        max_iterations = 2,
    )
    mktempdir() do dir
        out = joinpath(dir, "energy-only")
        r = fixed_spectrum(state, v; spectrum = o, output = out)
        @test r.converged && last(r.history).iteration == 2
        @test last(r.history).max_residual > 1e-7
        @test last(r.history).occupied_band_energy ≈
              dot(p.occupations, last(r.history).eigenvalues[1:2])
        status = K.JSON3.read(read(joinpath(out, "status.json"), String))
        @test status.criterion == "energy" && status.converged
        @test !haskey(status, :residual_converged)
        @test haskey(status.final, :max_residual) && haskey(status.final, :gram_defect)
        stages = K.JSON3.read.(readlines(joinpath(out, "stages.jsonl")))
        calls = K.JSON3.read.(readlines(joinpath(out, "tci_calls.jsonl")))
        @test count(r -> r.stage == "green_rhs", stages) == o.states * last(r.history).iteration
        @test count(r -> r.stage == "green_mpo", stages) == o.states * last(r.history).iteration
        @test sum(row.tci_calls for row in stages) == length(calls)
        @test all(row -> row.seconds >= 0 && haskey(row, :parent_stage), stages)
        @test all(
            row ->
                row.post_rounding_seconds >= 0 &&
                    row.interpolation_and_rounding_seconds ≈
                    row.seconds + row.post_rounding_seconds,
            calls,
        )
    end
    @test !K.spectrum_accepted(o, 1, 0.0)
    @test K.spectrum_accepted(o, 2, 0.0)
    @test !K.spectrum_accepted(o, 2, nothing)
    @test !K.spectrum_accepted(o, 2, NaN)
    @test !K.spectrum_accepted(o, 2, Inf)
    @test !K.spectrum_accepted(o, 2, -1.0)
    @test !K.spectrum_accepted(o, 2, o.energy_tolerance)
    @test all(
        k -> !(k in fieldnames(FixedSpectrumOptions)),
        (:residual_tolerance, :gram_tolerance, :criterion),
    )
    @test_throws ArgumentError K.options_from_dict(
        FixedSpectrumOptions,
        Dict("criterion" => "residual"),
    )
end
end
