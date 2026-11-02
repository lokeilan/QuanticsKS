module PivotSamplingChecks
using Test, QuanticsKS, LinearAlgebra, Random, JSON3, Serialization
const K = QuanticsKS
const Q = K.Q
const rng = MersenneTwister(20261102)
values(x) = [only(prod(g[:, ((j >> (k - 1)) & 1) + 1, :]
                       for (k, g) in enumerate(x.c))) for j = 0:(2^length(x) - 1)]
function random_train(R, r = 3)
    ranks = [1; fill(r, R - 1); 1]
    Q.normalize(Q.QTT([randn(rng, ComplexF64, ranks[k], 2, ranks[k + 1]) for k = 1:R]))
end
function expected_slice(v, R, left, right, ::Val{M}) where M
    dense = reshape(v, ntuple(_ -> 2, R))
    out = zeros(ComplexF64, length(left), ntuple(_ -> 2, M)..., length(right))
    for ci in CartesianIndices(out)
        q = vcat(left[ci[1]], collect(Tuple(ci)[2:end-1]), right[ci[M + 2]])
        out[ci] = dense[q...]
    end
    out
end

@testset "Fit-local caches eliminate repeated scalar and slice evaluation" begin
    R = 16
    counts = zeros(Int, 1 << R)
    opts = K.option_tuple(TCIOptions(scalar_cache_capacity = 7))
    source = Q.sampled(j -> (counts[j + 1] += 1; complex(j, -j)), R)
    oracle = Q.RecordedSamples(source, opts)
    for j = 0:((1 << R) - 1)
        oracle(j)
    end
    @test all(oracle(j) == complex(j, -j) for j = ((1 << R) - 1):-1:0)
    @test all(==(1), counts)
    @test oracle.evaluations == 1 << R
    @test oracle.cache.evictions == 0
    @test length(oracle.cache.values) == 1 << R

    counts = zeros(Int, 64)
    source = Q.sampled(j -> (counts[j + 1] += 1; complex(j + 1, j)), 6)
    oracle = Q.RecordedSamples(source, opts)
    left, right = Q.bits.(0:3, 2), Q.bits.(0:3, 2)
    oracle(0)
    oracle(31)
    got = oracle(left, right, Val(2))
    expected = expected_slice(complex.(1:64, 0:63), 6, left, right, Val(2))
    @test got == expected
    @test oracle(left, right, Val(2)) == expected
    @test all(==(1), counts)
    @test oracle.evaluations == 64
    @test oracle.computed_batches > 0
    @test_throws BoundsError oracle(64)

    sparse = Q.RecordedSamples(source, merge(opts, (; dense_cache_max_points = 0)))
    for j = 0:63
        @test sparse(j) == complex(j + 1, j)
        @test length(sparse.cache) <= 7
    end
    @test sparse.cache.evictions == 57
    @test length(sparse.cache) == 7
end

@testset "Native cross slices and periodic carries match independent dense values" begin
    R = 7
    x, y = random_train(R), random_train(R, 2)
    saved = deepcopy(x.c)
    rec = K.recorder(nothing, TCIOptions(environment_cache_bits = 2, slice_cache_capacity = 17))
    sx, sy = Q.sampled(x, rec), Q.sampled(y, rec)
    vx, vy = values(x), values(y)
    expressions = (sx, Q.pointwise((a, b) -> 2a - im*b + a*b, sx, sy),
                   Q.shifted(sx, -1), Q.shifted(sx, 1), Q.shifted(sx, 129))
    targets = (vx, 2vx - im*vy + vx.*vy, circshift(vx, 1), circshift(vx, -1), circshift(vx, -129))
    for M in (0, 1, 2), nl in unique((0, 2, R - M))
        nr = R - M - nl
        left = Q.bits.(unique([0, (1 << nl) - 1, min(1, (1 << nl) - 1)]), nl)
        right = Q.bits.(unique([0, (1 << nr) - 1, min(2, (1 << nr) - 1)]), nr)
        for (f, target) in zip(expressions, targets)
            @test f(left, right, Val(M)) ≈ expected_slice(target, R, left, right, Val(M)) atol = 2e-13
        end
    end
    @test x.c == saved
    @test sum(length, sx.frames.cacheleft) + sum(length, sx.frames.cacheright) <= 17
    @test isempty(sx(Vector{Int}[], [Int[]], Val(1)))
    oracle = Q.block_oracle([x, y], rec)
    l, r = Q.bits.(0:3, 2), Q.bits.(0:7, 3)
    f = Q.LinearSamples(oracle, ComplexF64[2, -im])
    @test f(l, r, Val(2)) ≈ expected_slice(2vx - im*vy, R, l, r, Val(2)) atol = 2e-13
    d = Q.DensitySamples(oracle, [1.0, 2.0])
    @test d(l, r, Val(2)) ≈ expected_slice(abs2.(vx) + 2abs2.(vy), R, l, r, Val(2)) atol = 2e-13
end

@testset "Full geometry survives iterations and reloads while values change" begin
    mktempdir() do dir
        options = TCIOptions(initial_random_pivots = 3, environment_cache_bits = 2)
        rec = K.recorder(dir, options)
        x, y = random_train(6, 2), random_train(6, 2)
        vx, vy = values(x), values(y)
        firstfit = Q.cross(rec, 6, Q.sampled(x, rec); tol = 1e-11, cap = 16,
                           label = "persistent", shared_pivots = "group")
        @test values(firstfit) ≈ vx atol = 1e-10
        rec.scf = 2
        changed = Q.pointwise((a, b) -> 2a + im*b, Q.sampled(x, rec), Q.sampled(y, rec))
        secondfit = Q.cross(rec, 6, changed; tol = 1e-11, cap = 16, label = "persistent")
        @test values(secondfit) ≈ 2vx + im*vy atol = 1e-10
        grouped = Q.cross(rec, 6, Q.sampled(y, rec); tol = 1e-11, cap = 16,
                          label = "another_orbital", shared_pivots = "group")
        @test values(grouped) ≈ vy atol = 1e-10
        @test rec.cross_states[("persistent", 6)].uses == 2
        Q.save_cross_state(rec, joinpath(dir, "pivots.jls"))
        restored = K.recorder(nothing, options)
        Q.load_cross_state!(restored, joinpath(dir, "pivots.jls"))
        @test restored.cross_states[("persistent", 6)].left == rec.cross_states[("persistent", 6)].left
        @test restored.cross_states[("persistent", 6)].right == rec.cross_states[("persistent", 6)].right
        thirdfit = Q.cross(restored, 6, Q.sampled(y, restored); tol = 1e-11, cap = 16, label = "persistent")
        @test values(thirdfit) ≈ vy atol = 1e-10
        close(rec)
        rows = JSON3.read.(readlines(joinpath(dir, "tci_calls.jsonl")))
        @test rows[2].pivot_reused
        @test rows[2].pivot_source == "previous_fit"
        @test rows[3].pivot_source == "orbital_group"
        @test all(row -> row.native_batch && row.batch_calls > 0, rows)
        @test all(row -> row.oracle_evaluations <= 64 && row.cache_evictions == 0, rows)
        @test isfile(joinpath(dir, "cross-state.jls"))
        bad = deserialize(joinpath(dir, "pivots.jls"))
        bad["states"][1]["left"][1] = [[2]]
        serialize(joinpath(dir, "bad.jls"), bad)
        @test_throws ArgumentError Q.load_cross_state!(restored, joinpath(dir, "bad.jls"))
    end
end

@testset "Orthogonalization and Ritz materialize once and preserve the occupied space" begin
    mktempdir() do dir
        R = 6
        p = Problem(PeriodicGrid1D(R, 8.0), LocalPotential1D(0.0), ones(3))
        cfg = K.solver_config(p, Accuracy(orbital_cap = 32, work_cap = 64,
            orthogonalization_tolerance = 1e-12), SCFOptions(orthogonalize_threshold = 1e-9))
        rec = K.recorder(dir, TCIOptions(environment_cache_bits = 3))
        xs = [random_train(R, 2) for _ = 1:3]
        xs[2] = Q.normalize(Q.add(xs[1], xs[2]; b = 0.7, tol = 0.0, cap = 32))
        saved = deepcopy([x.c for x in xs])
        X = hcat(values.(xs)...)
        v = Q.cross(rec, R, j -> 0.3cos(2pi*j/2^R); tol = 1e-13, cap = 16)
        H = Q.Hamiltonian(v, Q.diagonal(v), Q.circulant(Q.delta_tt(R, 1)),
                          Q.circulant(Q.delta_tt(R, 2^R - 1)), K.spacing(p.grid))
        A = Matrix(Diagonal(values(v) .+ 1/H.h^2))
        for j = 1:2^R, offset in (-1, 1)
            A[j, mod1(j + offset, 2^R)] -= 1/(2H.h^2)
        end
        S = eigen(Hermitian(X' * X))
        B = X * S.vectors * Diagonal(1 ./ sqrt.(S.values)) * S.vectors'
        before = rec.calls
        ys, defect, offdiag = Q.orthogonalize_ritz(xs, H, cfg, rec)
        Y = hcat(values.(ys)...)
        @test rec.calls - before == length(xs)
        @test defect < 1e-9
        @test offdiag > 0
        @test Y' * Y ≈ I atol = 2e-10
        @test Y * Y' ≈ B * B' atol = 2e-10
        @test real.(diag(Y' * A * Y)) ≈ eigvals(Hermitian(B' * A * B)) atol = 2e-8
        @test norm(Y' * A * Y - Diagonal(diag(Y' * A * Y))) < 2e-8
        @test [x.c for x in xs] == saved
        close(rec)
        events = JSON3.read.(readlines(joinpath(dir, "stages.jsonl")))
        event = only(filter(e -> get(e, :event, "") == "orbital_transform", events))
        @test event.coefficient_transforms >= 2
        @test event.materialization_passes == 1
        @test !event.post_fit_orthogonalization
    end
end
end
