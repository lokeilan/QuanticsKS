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
    Q.scale(x, inv(sqrt(real(Q.inner_tci(x, x)))))
end

function dense(x)
    p = size(first(x.c), 2)
    [
        only(prod(g[:, mod(div(j, p^(k - 1)), p) + 1, :] for (k, g) in enumerate(x.c))) for
        j = 0:(p^length(x) - 1)
    ]
end

struct ObservedScratch <: AbstractVector{Vector{ComplexF64}}
    data::Vector{Vector{ComplexF64}}
    tasks::Vector{Any}
end
Base.size(s::ObservedScratch) = size(s.data)
function Base.getindex(s::ObservedScratch, i::Int)
    s.tasks[i] = (; task = current_task(), pool = Threads.threadpool())
    s.data[i]
end

@testset "Single direct inner uses bounded internal workers" begin
    cfg = (; parallel_tasks = 0)
    for cap in (16, 32, 48, 64)
        x, y = train(16, cap), train(16, cap - 1)
        saved = deepcopy((x.c, y.c))
        reference = Q.inner(x, y)
        @test Q.inner(x, y; cfg) ≈ reference atol = 2e-13
        @test Q.inner(y, x; cfg) ≈ conj(reference) atol = 2e-13
        @test Q.ttnorm(x; cfg) ≈ Q.ttnorm(x) atol = 2e-13
        @test (x.c, y.c) == saved
        nested = Q.independent_map(Any, 4, cfg) do i
            (; value = Q.inner(x, y; cfg), workers = Q.inner_site_workers(x.c[8], y.c[8], cfg))
        end
        @test all(r -> r.value ≈ reference, nested)
        if Threads.nthreads(:default) > 1
            @test all(r -> r.workers == 1, nested)
        end
    end
    large_x, large_y = train(16, 48), train(16, 48)
    @test Q.inner_direct_workers(large_x, large_y, cfg) == min(Threads.nthreads(:default), 48)
    @test Q.inner_direct_workers(train(16, 32), train(16, 32), cfg) == 1
    gx = randn(rng, ComplexF64, 48, 2, 47)
    gy = randn(rng, ComplexF64, 44, 2, 43)
    input = randn(rng, ComplexF64, 48 * 44)
    expected = zeros(ComplexF64, 47 * 43)
    Q.inner_direct_site!(
        expected,
        input,
        gx,
        gy,
        [zeros(ComplexF64, 2 * 44)],
        (; parallel_tasks = 1),
    )
    for requested in (0, 2)
        localcfg = (; parallel_tasks = requested)
        workers = Q.inner_site_workers(gx, gy, localcfg)
        scratch = ObservedScratch(
            [zeros(ComplexF64, 2 * 44) for _ = 1:workers],
            Vector{Any}(undef, workers),
        )
        actual = zeros(ComplexF64, 47 * 43)
        Q.inner_direct_site!(actual, input, gx, gy, scratch, localcfg)
        @test actual == expected
        @test length(unique(s.task for s in scratch.tasks)) == workers
        workers > 1 && @test all(s -> s.pool == :default, scratch.tasks)
    end
    @test Q.inner_site_workers(gx, gy, (; parallel_tasks = 1)) == 1
    @test Q.inner_site_workers(zeros(ComplexF64, 2, 2, 2), zeros(ComplexF64, 2, 2, 2), cfg) == 1
end

@testset "Conditional inner contraction and independent calls" begin
    for physical in (2, 3), R in (1, 2, 5)
        x, y = train(R, 3; physical), train(R, 5; physical)
        saved = deepcopy((x.c, y.c))
        @test Q.inner(x, y) ≈ dot(dense(x), dense(y)) atol = 2e-13
        @test Q.inner(x, y) ≈ conj(Q.inner(y, x)) atol = 2e-13
        @test (x.c, y.c) == saved
    end
    pairs = [(train(16, cap), train(16, cap)) for cap in (4, 16, 48, 64, 96)]
    @test Q.inner_backend(pairs[3]...) == :direct
    @test Q.inner_backend(pairs[4]...) == :tci
    @test Q.inner_backend(pairs[5]...) == :tci
    push!(pairs, (last(pairs)[1], train(16, 1)))
    @test Q.inner_backend(last(pairs)...) == :direct
    expected = [Q.inner_tci(x, y) for (x, y) in pairs]
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
