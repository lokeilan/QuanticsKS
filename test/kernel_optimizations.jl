module KernelOptimizationChecks

using Test, QuanticsKS, LinearAlgebra, Random
const Q = QuanticsKS.Q
const rng = MersenneTwister(20261104)
BLAS.set_num_threads(1)

function train(R, cap; physical = 2)
    bonds = [min(physical^min(k, R - k), cap) for k = 0:R]
    x = Q.QTT([
        randn(rng, ComplexF64, bonds[k], physical, bonds[k + 1]) / sqrt(physical * bonds[k]) for
        k = 1:R
    ])
    Q.scale(x, inv(sqrt(real(Q.inner(x, x)))))
end

function dense(x)
    p = size(first(x.c), 2)
    [
        only(prod(g[:, mod(div(j, p^(k - 1)), p) + 1, :] for (k, g) in enumerate(x.c))) for
        j = 0:(p^length(x) - 1)
    ]
end

@testset "Library inner contraction and independent calls" begin
    for physical in (2, 3), R in (1, 2, 5)
        x, y = train(R, 3; physical), train(R, 5; physical)
        saved = deepcopy((x.c, y.c))
        @test Q.inner(x, y) ≈ dot(dense(x), dense(y)) atol = 2e-13
        @test Q.inner(x, y) ≈ conj(Q.inner(y, x)) atol = 2e-13
        @test (x.c, y.c) == saved
    end
    pairs = [(train(16, cap), train(16, cap)) for cap in (4, 16, 48, 64, 96)]
    push!(pairs, (last(pairs)[1], train(16, 1)))
    expected = [Q.inner(x, y) for (x, y) in pairs]
    actual = Q.independent_map(
        i -> Q.inner(pairs[i]...),
        ComplexF64,
        length(pairs),
        (; parallel_tasks = 0),
    )
    @test actual ≈ expected atol = 2e-13
    x = train(5, 3)
    @test_throws DimensionMismatch Q.inner(x, train(4, 3))
    @test_throws DimensionMismatch Q.inner(x, train(5, 3; physical = 3))
    bad = deepcopy(x.c)
    bad[3] = zeros(ComplexF64, size(bad[3], 1) + 1, 2, size(bad[3], 3))
    @test_throws DimensionMismatch Q.inner(x, Q.QTT(bad))
end

@testset "L4 projection uses the existing tangent gauge" begin
    for R in (5, 8)
        x = train(R, 4)
        frame = Q.canonical_frame(x; cfg = (; parallel_tasks = 0))
        raw = hcat([Q.project(frame, train(R, 6)) for _ = 1:5]...)
        fac = svd(raw)
        D = fac.U[:, fac.S .> maximum(fac.S) * 1e-7]
        v = train(R, 3)
        H = Q.Hamiltonian(v, Q.diagonal(v), Q.shift_mpo(R, 1), Q.shift_mpo(R, -1), 0.075)
        saved = deepcopy((frame.left, frame.right, x.c, D))
        reference = hcat([Q.h_project(frame, Q.tangent_tt(frame, z), H) for z in eachcol(D)]...)
        actual = Q.independent_columns(size(D, 1), size(D, 2), (; parallel_tasks = 0)) do j
            Q.h_project_tangent(frame, view(D, :, j), H)
        end
        @test actual ≈ reference rtol = 2e-12 atol = 2e-11
        @test D' * actual ≈ D' * reference rtol = 2e-12 atol = 2e-11
        @test norm(Q.project(frame, Q.tangent_tt(frame, D[:, 1])) - D[:, 1]) < 2e-13
        @test (frame.left, frame.right, x.c, D) == saved
    end
end

end
