module EvaluateWorkTensorChecks
using Test, QuanticsKS, LinearAlgebra, Random, JSON3
const K = QuanticsKS
const Q = K.Q
BLAS.set_num_threads(1)
const rng = MersenneTwister(20261102)
values(x) = [only(prod(g[:, ((j >> (k - 1)) & 1) + 1, :]
                       for (k, g) in enumerate(x.c))) for j = 0:(2^length(x) - 1)]
function random_train(R, r)
    ranks = [1; fill(r, R - 1); 1]
    Q.normalize(Q.QTT([randn(rng, ComplexF64, ranks[k], 2, ranks[k + 1]) for k = 1:R]))
end
function dense_hamiltonian(H)
    v = values(H.potential)
    A = Matrix(Diagonal(v .+ 1 / H.h^2))
    for j in eachindex(v), step in (-1, 1)
        A[j, mod1(j + step, length(v))] -= 1 / (2H.h^2)
    end
    A
end
function settings(R)
    p = Problem(PeriodicGrid1D(R, 8.0), LocalPotential1D(0.0), [1.0, 1.0]; mass = 1.7)
    a = Accuracy(orbital_cap = 32, work_cap = 64, field_cap = 64,
                 algebra_tolerance = 1e-13, orthogonalization_tolerance = 1e-12)
    p, a, K.solver_config(p, a)
end

@testset "Evaluate final work tensors against independent dense FD2" begin
    for R in (3, 7)
        p, a, cfg = settings(R)
        rec = K.recorder(nothing, TCIOptions(environment_cache_bits = 3))
        v = Q.cross(rec, R, j -> 0.3cos(2pi*j/2^R) + 0.1sin(4pi*j/2^R); tol = 1e-13, cap = 32)
        H = Q.Hamiltonian(v, Q.diagonal(v), Q.circulant(Q.delta_tt(R, 1)),
                          Q.circulant(Q.delta_tt(R, 2^R - 1)), K.spacing(p.grid) * sqrt(p.mass))
        A = dense_hamiltonian(H)
        xs = [random_train(R, r) for r in (2, 3)]
        X = hcat(values.(xs)...)
        calls = rec.calls
        @test Q.h_gram(xs, H) ≈ X' * A * X atol = 2e-11 rtol = 1e-11
        @test rec.calls == calls
        for (i, x) in enumerate(xs)
            ex = Q.h_evaluator(H, x, rec)
            @test [ex(j) for j = 0:2^R-1] ≈ A * X[:, i] atol = 2e-12
            energy = real(dot(X[:, i], A * X[:, i]))
            @test Q.h_expectation(H, x) ≈ energy atol = 2e-11
            saved = deepcopy(x.c)
            calls = rec.calls
            residual = Q.h_residual(H, x, energy, cfg, rec; tol = 1e-12)
            @test rec.calls == calls + 1
            @test values(residual) ≈ (A - energy * I) * X[:, i] atol = 2e-10 rtol = 1e-9
            @test x.c == saved
            calls = rec.calls
            action = Q.h_action(H, x, cfg, rec; work_tol = 0.0)
            @test rec.calls == calls + 1
            @test values(action) ≈ A * X[:, i] atol = 2e-10
            green = Q.green_mpo(R, H.h, 0.5)
            frame = Q.canonical_frame(xs[1])
            y = Q.apply_mpo(green, residual; tol = 1e-13, cap = 64)
            @test Q.project(frame, residual, green) ≈ Q.project(frame, y) atol = 2e-10 rtol = 1e-9
            shifted = Q.apply_mpo(H.plus, x; tol = 1e-13, cap = 32)
            calls = rec.calls
            @test Q.difference_norm(shifted, x; rec) ≈ norm(circshift(X[:, i], 1) - X[:, i]) atol = 2e-11
            @test rec.calls == calls
        end
        @test_throws ArgumentError Q.h_residual(H, xs[1], NaN, cfg, rec)
        close(rec)
    end
end

@testset "Small residual and near-constant kinetic energy do not use large-norm subtraction" begin
    R = 16
    zero_field = Q.constant_tt(R, 0)
    H = Q.Hamiltonian(zero_field, Q.diagonal(zero_field),
                      Q.circulant(Q.delta_tt(R, 1)), Q.circulant(Q.delta_tt(R, 2^R - 1)), 8.0 / 2^R)
    x0 = Q.normalize(Q.constant_tt(R))
    epsilon = 1e-7
    x = Q.normalize(Q.add(x0, Q.phase_tt(x0, 1); b = epsilon, tol = 0.0, cap = 4))
    expected = epsilon^2 / (1 + epsilon^2) * 2sin(pi / 2^R)^2 / H.h^2
    @test Q.h_expectation(H, x) ≈ expected rtol = 2e-5 atol = 1e-19
    @test abs(Q.h_expectation(H, x0)) < 1e-19
end

@testset "Final residual fits and fixed-H targets" begin
    p, a, cfg = settings(5)
    rec = K.recorder(nothing, TCIOptions())
    x = random_train(5, 2)
    v = Q.constant_tt(5, 0.0)
    H = Q.Hamiltonian(v, Q.diagonal(v), Q.circulant(Q.delta_tt(5, 1)),
                      Q.circulant(Q.delta_tt(5, 31)), K.spacing(p.grid) * sqrt(p.mass))
    potential = [0.2cos(2pi*j/32) for j = 0:31]
    options = FixedSpectrumOptions(states = 2, occupied_states = 2, virtual_wavevectors = Int[],
                                   projection = :qtt_gradient, diagnostics = :qtt)
    projected = K.spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = 1e-13)
    A = dense_hamiltonian(H) + Diagonal(potential)
    X = values(x)
    before = rec.calls
    energies, residuals = K.spectrum_residuals([x], H, cfg, options, rec, potential)
    energy = real(dot(X, A * X))
    @test energies[1] ≈ energy atol = 2e-10
    @test residuals[1] ≈ norm(A * X - energy * X) / norm(X) atol = 2e-9
    @test rec.calls == before + 2 # One shared potential and one final residual.
    @test values(projected.potential) ≈ potential atol = 1e-12
    before = rec.calls
    explicit_action = only(K.spectrum_actions([x], H, cfg, rec, options, potential))
    @test values(explicit_action) ≈ A * X atol = 2e-10
    @test rec.calls == before + 2
    # A zero initial residual oracle is handled by core algebra, not sampled-zero inference.
    zero = Q.h_residual(H, Q.constant_tt(5), 0, cfg, rec; tol = 1e-13)
    @test norm(values(zero)) < 2e-10
    close(rec)
end

@testset "Known fixed potential supplies an informative sparse pivot" begin
    R = 16
    p, a, cfg = settings(R)
    rec = K.recorder(nothing, TCIOptions(initial_random_pivots = 0))
    v = Q.constant_tt(R, 0)
    H = Q.Hamiltonian(v, Q.diagonal(v), Q.circulant(Q.delta_tt(R, 1)),
                      Q.circulant(Q.delta_tt(R, 2^R - 1)), K.spacing(p.grid) * sqrt(p.mass))
    potential = zeros(2^R)
    potential[10038] = 0.2
    projected = K.spectrum_projection_hamiltonian(H, potential, cfg, rec; tol = 1e-12)
    ev = Q.evaluate(projected.potential, rec)
    @test ev(10037) ≈ 0.2 atol = 1e-12
    @test abs(ev(0)) < 1e-12
    close(rec)
end
end
