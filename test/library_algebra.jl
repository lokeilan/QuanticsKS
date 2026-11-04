module LibraryAlgebraChecks

using Test, QuanticsKS, LinearAlgebra, Random, JSON3
const K = QuanticsKS
const Q = K.Q
const rng = MersenneTwister(411)

function random_train(R; rank = 2)
    ranks = [1; fill(rank, R - 1); 1]
    Q.QTT([randn(rng, ComplexF64, ranks[k], 2, ranks[k + 1]) for k = 1:R])
end
values(x) = [
    only(prod(g[:, ((j >> (k - 1)) & 1) + 1, :] for (k, g) in enumerate(x.c))) for
    j = 0:(2^length(x) - 1)
]

@testset "Library complex inner products against dense contraction" begin
    for R in (1, 2, 5, 6)
        x, y = random_train(R; rank = 2), random_train(R; rank = 3)
        saved_x, saved_y = deepcopy(x.c), deepcopy(y.c)
        vx, vy = values(x), values(y)
        @test Q.inner(x, y) ≈ dot(vx, vy)
        @test Q.inner(y, x) ≈ conj(Q.inner(x, y))
        @test Q.inner(Q.scale(x, 2 + im), y) ≈ (2 - im) * dot(vx, vy)
        @test Q.ttnorm(x) ≈ norm(vx)
        @test iszero(Q.inner(Q.constant_tt(R, 0), y))
        @test iszero(Q.inner(Q.delta_tt(R, 0), Q.delta_tt(R, 1)))
        @test Q.gram([x, y]) ≈ hcat(vx, vy)' * hcat(vx, vy)
        @test Q.gram([x], [x, y]) ≈ reshape(vx, :, 1)' * hcat(vx, vy)
        @test x.c == saved_x
        @test y.c == saved_y
    end
end

@testset "Library TT adapters and bounded point evaluation" begin
    x = random_train(5)
    saved = deepcopy(x.c)
    v = values(x)
    @test values(Q.scale(x, 2 - im)) ≈ (2 - im) .* v
    @test values(Q.rounded(x; tol = 1e-10, cap = 16)) ≈ v
    @test Q.rank(Q.rounded(x; tol = 0.0, cap = 1)) == 1
    @test x.c == saved
    for cachebits in (0, 1, 10)
        evaluator = Q.Evaluator(x; cachebits)
        for j in vcat(collect(0:31), collect(31:-1:0))
            @test evaluator(j) ≈ v[j + 1]
        end
        @test size(evaluator.left, 2) <= 2^cachebits
        @test size(evaluator.right, 2) <= 2^cachebits
        @test_throws BoundsError evaluator(-1)
        @test_throws BoundsError evaluator(32)
    end
    @test_throws ArgumentError Q.rounded(x; cap = 0)
    @test_throws ArgumentError Q.rounded(x; tol = -1)
    @test_throws DimensionMismatch Q.inner(x, random_train(4))
end

function sampled_sum(evaluator, points)
    total = 0.0 + 0.0im
    for j in points
        total += evaluator(j)
    end
    total
end

@testset "Endpoint environment cache semantics and allocation-free queries" begin
    # Odd and long trains cover the uncached middle-core path; cachebits=0
    # must use only scratch vectors, with no environment expansion.
    for R in (1, 16, 17, 22), cachebits in (0, 8)
        x = random_train(R; rank = 3)
        saved = deepcopy(x.c)
        evaluator = Q.Evaluator(x; cachebits)
        points = unique([0, 2^R - 1, 1, 2^(R - 1), rand(rng, 0:(2^R - 1), 12)...])
        expected = [
            only(prod(g[:, ((j >> (k - 1)) & 1) + 1, :] for (k, g) in enumerate(x.c))) for
            j in points
        ]
        @test evaluator.(points) ≈ expected
        @test evaluator.(reverse(points)) ≈ reverse(expected)
        @test size(evaluator.left, 2) <= 2^cachebits
        @test size(evaluator.right, 2) <= 2^cachebits
        @test x.c == saved
        sampled_sum(evaluator, points) # Compile before checking hot-query allocations.
        @test (@allocated sampled_sum(evaluator, points)) == 0
    end
    x = random_train(16; rank = 3)
    evaluator = Q.Evaluator(x; cachebits = 8)
    @test evaluator.prefix == evaluator.suffix == 8
    @test size(evaluator.left, 2) == size(evaluator.right, 2) == 256
    left, right = copy(evaluator.left), copy(evaluator.right)
    sampled_sum(evaluator, rand(rng, 0:65535, 10000))
    @test evaluator.left == left && evaluator.right == right
    @test_throws ArgumentError Q.Evaluator(x; cachebits = -1)
    @test_throws ArgumentError Q.Evaluator(x; cachebits = 17)
end

@testset "Pointwise sums and a single reconstruction for multiple terms" begin
    x, y, z = [random_train(4) for _ = 1:3]
    rec = K.recorder(nothing, TCIOptions(initial_random_pivots = 0))
    vx, vy, vz = values.((x, y, z))
    calls = rec.calls
    result = Q.linear_combination(rec, (x, y, z), (2, -im, -0.5); tol = 1e-10, cap = 16)
    @test rec.calls == calls + 1
    @test values(result) ≈ 2vx - im * vy - 0.5vz rtol = 1e-8
    oracle = Q.block_oracle((x, y, z), rec)
    shared = Q.linear_combination(rec, (x, y, z), (2, 0, -0.5); oracle, cap = 16)
    @test values(shared) ≈ 2vx - 0.5vz rtol = 1e-8
    @test values(Q.add(x, y; a = im, b = 2, rec, cap = 16)) ≈ im * vx + 2vy rtol = 1e-8
    @test values(Q.add(x, y; a = im, b = 2, cap = 16)) ≈ im * vx + 2vy
    @test norm(values(Q.add(x, x; b = -1, rec, cap = 16))) < 1e-8
    @test iszero(Q.ttnorm(Q.linear_combination(rec, (x, y), (0, 0))))
    @test_throws DimensionMismatch Q.add(x, random_train(3); rec)
    @test_throws DimensionMismatch Q.linear_combination(rec, (x, y), (1,))
    @test_throws ArgumentError Q.linear_combination(rec, (x,), (NaN,))

    # A single-site train needs no cross-interpolation sweep.
    a, b = random_train(1), random_train(1)
    @test values(Q.add(a, b; a = 2, b = -3, rec)) ≈ 2values(a) - 3values(b)

    # Zero initial samples do not prove a globally zero function.
    mktempdir() do dir
        recorder = K.recorder(dir, TCIOptions(initial_random_pivots = 0, zero_check_limit = 16))
        try
            localized = Q.delta_tt(14, 10037)
            result = Q.add(localized, Q.constant_tt(14, 0.0); rec = recorder, cap = 4)
            @test Q.Evaluator(result)(10037) ≈ 1
            @test recorder.samples == 2
        finally
            close(recorder)
        end
        row = JSON3.read(only(readlines(joinpath(dir, "tci_calls.jsonl"))))
        @test row.construction == "library_tt_add_zero_initial_pivots"
        @test !haskey(row, :verified_zero)
    end
end

@testset "Library MPO application and canonical frames" begin
    x = random_train(5)
    saved = deepcopy(x.c)
    v = values(x)
    shift = Q.circulant(Q.delta_tt(5, 1))
    for algorithm in (:zipup, :naive)
        result = Q.apply_mpo(shift, x; tol = 1e-10, cap = 16, algorithm)
        @test values(result) ≈ circshift(v, 1) rtol = 1e-8
    end
    field = random_train(5)
    @test values(Q.apply_mpo(Q.diagonal(field), x; cap = 32)) ≈ values(field) .* v rtol = 1e-8
    @test_throws DimensionMismatch Q.apply_mpo(shift[1:4], x)
    frame = Q.canonical_frame(x)
    @test values(Q.QTT(frame.left)) ≈ v
    @test values(Q.QTT(frame.right)) ≈ v
    for g in frame.left[1:(end - 1)]
        m = reshape(g, :, size(g, 3))
        @test m' * m ≈ I
    end
    for g in frame.right[2:end]
        m = reshape(g, size(g, 1), :)
        @test m * m' ≈ I
    end
    @test all(isfinite, Q.project(frame, x))
    @test x.c == saved
end

@testset "SVD MPO contraction boundaries" begin
    for R in (1, 2, 6)
        x = random_train(R; rank = 3)
        field = random_train(R; rank = 2)
        w = Q.diagonal(field)
        saved_x, saved_w = deepcopy(x.c), deepcopy(w)
        product = Q.apply_mpo(w, x; tol = 1e-12, cap = 16)
        @test values(product) ≈ values(field) .* values(x) rtol = 1e-9
        @test Q.rank(product) <= 16
        @test x.c == saved_x
        @test w == saved_w
        real_operator = real.(Q.diagonal(Q.constant_tt(R, 2)))
        @test values(Q.apply_mpo(real_operator, x; cap = 16)) ≈ 2values(x)
        zero_product = Q.apply_mpo(w, Q.constant_tt(R, 0); cap = 4)
        @test all(g -> all(isfinite, g), zero_product.c)
        @test all(iszero, values(zero_product))
        disjoint = Q.apply_mpo(Q.diagonal(Q.delta_tt(R, 0)), Q.delta_tt(R, 1))
        @test all(iszero, values(disjoint))
        @test all(g -> all(isfinite, g), disjoint.c)
        exact = Q.apply_mpo(real_operator, Q.constant_tt(R, 1); tol = 0.0, cap = 16)
        @test values(exact) ≈ fill(2, 2^R)
        limited = Q.apply_mpo(w, x; cap = 1)
        @test Q.rank(limited) == 1
        @test all(g -> all(isfinite, g), limited.c)
    end
end

end
