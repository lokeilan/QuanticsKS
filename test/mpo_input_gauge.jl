module MPOInputGaugeChecks
using Test, QuanticsKS, LinearAlgebra, Random
const Q = QuanticsKS.Q
const rng = MersenneTwister(20261104)
function values(x)
    sample = Q.Evaluator(x; cachebits = 8)
    [sample(j) for j = 0:(2^length(x) - 1)]
end

function regauge(x)
    cores = deepcopy(x.c)
    for k = 1:(length(x) - 1)
        r = size(cores[k], 3)
        factors = r == 1 ? [1.0] : 10.0 .^ range(-3, 3; length = r)
        for j in eachindex(factors)
            cores[k][:, :, j] .*= factors[j]
            cores[k + 1][j, :, :] ./= factors[j]
        end
    end
    Q.QTT(cores)
end

@testset "Consumer QR preserves values and establishes a right suffix" begin
    for R in (1, 2, 7), zero in (false, true)
        ranks = [1; fill(3, R - 1); 1]
        x =
            zero ? Q.constant_tt(R, 0) :
            Q.QTT([randn(rng, ComplexF64, ranks[k], 2, ranks[k + 1]) for k = 1:R])
        x = regauge(x)
        saved = deepcopy(x.c)
        canonical = Q.right_canonical(x)
        @test norm(values(canonical) - values(x)) <= 1e-11 * max(norm(values(x)), 1)
        @test x.c == saved
        for g in canonical.c[2:end]
            A = reshape(g, size(g, 1), :)
            @test norm(A * A' - I) < 2e-12
        end
    end
end

@testset "Green action is stable under equivalent input gauges" begin
    R, N, cell, mu = 8, 256, 8.0, 0.5
    h = cell / N
    rec = QuanticsKS.recorder(nothing, TCIOptions())
    rhs = Q.cross(
        rec,
        R,
        j -> exp(0.2cos(2pi * j / N)) + 0.1im * sin(6pi * j / N);
        tol = 1e-12,
        cap = 16,
    )
    changed = regauge(rhs)
    green = Q.green_mpo(R, h, mu)
    shift = circshift(Matrix{Float64}(I, N, N), (1, 0))
    kinetic = (2I - shift - shift') / (2h^2)
    expected = (kinetic + mu * I) \ values(rhs)
    saved = deepcopy(changed.c)
    actual = Q.apply_mpo(green, changed; tol = 1e-12, cap = 16)
    @test norm(values(actual) - expected) / norm(expected) < 1e-8
    @test changed.c == saved
    canonical = Q.right_canonical(changed)
    explicit = Q.apply_mpo(green, canonical; tol = 1e-12, cap = 16, input_right_canonical = true)
    @test norm(values(explicit) - values(actual)) / norm(values(actual)) < 1e-10
    restored = Q.apply_mpo(green, rhs; tol = 1e-12, cap = 16, input_right_canonical = true)
    @test norm(values(restored) - expected) / norm(expected) < 1e-8
    @test norm(values(restored) - values(Q.apply_mpo(green, rhs; tol = 1e-12, cap = 16))) /
          norm(values(restored)) < 1e-10
    close(rec)
end
end
