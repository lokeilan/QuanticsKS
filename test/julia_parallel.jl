using Test, QuanticsKS, LinearAlgebra, Random, JSON3

const K = QuanticsKS
const Q = K.Q
BLAS.set_num_threads(1)

@testset "Bounded independent core work" begin
    cfg = (; parallel_tasks = 0)
    workers = min(4, Threads.nthreads(:default))
    @test Q.parallel_workers((; parallel_tasks = 4), 9) == workers
    @test Q.parallel_workers(cfg, 0) == 0
    @test Q.parallel_workers((;), 9) == 1
    @test_throws ArgumentError K.validate(SCFOptions(parallel_tasks = -1))
    @test_throws ArgumentError Q.foreach_independent(identity, 1, (; parallel_tasks = -1))
    @test size(Q.independent_columns(identity, 3, 0, cfg)) == (3, 0)

    seen = Vector{Any}(undef, 8)
    Q.with_svd_context(; stage = "parallel_context", orbital = 7) do
        Q.foreach_independent(8, (; parallel_tasks = 4)) do i
            sleep(0.005)
            seen[i] = (;
                task = current_task(),
                pool = Threads.threadpool(),
                context = task_local_storage()[Q.SVD_CONTEXT_KEY],
            )
        end
    end
    @test length(unique(r.task for r in seen)) == workers
    @test all(r -> r.context.stage == "parallel_context" && r.context.orbital == 7, seen)
    workers > 1 && @test all(r -> r.pool == :default, seen)
    @test !haskey(task_local_storage(), Q.SVD_CONTEXT_KEY)

    # Joining failures must wait for other workers before unwinding the caller.
    if workers > 1
        completed = Threads.Atomic{Int}(0)
        @test_throws CompositeException Q.foreach_independent(2, cfg) do i
            i == 1 && error("injected worker failure")
            sleep(0.02)
            Threads.atomic_add!(completed, 1)
        end
        @test completed[] == 1
    end

    Random.seed!(20261104)
    expected = rand(8)
    Random.seed!(20261104)
    Q.foreach_independent(i -> sum(abs2, 1:i), 8, cfg)
    @test rand(8) == expected
end

@testset "Serial and threaded QTT contractions" begin
    Random.seed!(20261104)
    p = Problem(
        PeriodicGrid1D(5, 8),
        LocalPotential1D(x -> 0.2cos(2pi * x / 8); id = "parallel-test"),
        [2.0, 2.0],
    )
    a = Accuracy(
        field_tolerance = 1e-11,
        work_tolerance = 1e-9,
        storage_tolerance = 1e-11,
        orbital_cap = 16,
        field_cap = 16,
        work_cap = 32,
    )
    xs =
        state_from_functions(
            p,
            [x -> 1 + 0.1cos(4pi * x / 8), x -> cis(2pi * x / 8) + 0.1cis(-4pi * x / 8)];
            accuracy = a,
        ).orbitals
    before = deepcopy(xs)
    rec = K.recorder(nothing, TCIOptions())
    ops = K.build_operators(p, a, rec, PhysicsNumerics())
    H = K.effective_hamiltonian(p, K.density(p, xs, a, rec), ops, a, rec)
    frame = Q.canonical_frame(first(xs))
    serial = K.solver_config(p, a, SCFOptions(parallel_tasks = 1))
    threaded = merge(serial, (; parallel_tasks = 0))
    n = frame.offsets[end]
    projected = Q.independent_columns(i -> Q.h_project(frame, xs[i], H), n, length(xs), threaded)
    @test projected ≈ hcat([Q.h_project(frame, x, H) for x in xs]...) rtol = 1e-13
    packed_serial, info_serial = Q.pack_orbitals(xs, serial; tol = a.storage_tolerance)
    packed_threaded, info_threaded = Q.pack_orbitals(xs, threaded; tol = a.storage_tolerance)
    @test info_serial == info_threaded
    @test all(i -> Q.difference_norm(packed_serial[i], packed_threaded[i]) < 1e-12, eachindex(xs))
    @test all(i -> xs[i].c == before[i].c, eachindex(xs))
    close(rec)
end

@testset "Threaded SCF preserves energy, orbitals, audit and logs" begin
    p = hydrogen_chain(R = 6, natoms = 4, distance = 4.0)
    a = Accuracy(
        orbital_cap = 16,
        field_cap = 32,
        work_cap = 32,
        field_tolerance = 1e-9,
        work_tolerance = 1e-8,
        storage_tolerance = 1e-10,
    )
    physics = PhysicsNumerics(xc_table_points = 64)
    tci = TCIOptions(global_pivot_search = 2, max_global_pivots = 2)
    mktempdir() do dir
        function solve(tasks, name)
            ground_state(
                p;
                accuracy = a,
                physics,
                tci,
                output = joinpath(dir, name),
                scf = SCFOptions(
                    parallel_tasks = tasks,
                    max_iterations = 3,
                    green_polish_every = 2,
                    green_polish_residual = 0.0,
                ),
            )
        end
        serial = solve(1, "serial")
        threaded = solve(0, "threaded")
        @test threaded.stop_reason == serial.stop_reason
        @test threaded.converged == serial.converged
        @test [r.total for r in threaded.history] ≈ [r.total for r in serial.history] atol = 1e-10
        @test [r.gram_defect for r in threaded.history] ≈ [r.gram_defect for r in serial.history] atol = 1e-10
        @test threaded.audit.max_residual ≈ serial.audit.max_residual atol = 1e-10
        @test threaded.audit.passed == serial.audit.passed
        for (x, y) in zip(serial.state.orbitals, threaded.state.orbitals)
            ex, ey = Q.Evaluator(x), Q.Evaluator(y)
            @test norm([ex(j) - ey(j) for j = 0:63]) < 1e-9
        end
        calls(name) = JSON3.read.(readlines(joinpath(dir, name, "tci_calls.jsonl")))
        @test [(r.label, r.oracle_evaluations) for r in calls("serial")] == [(r.label, r.oracle_evaluations) for r in calls("threaded")]
        stages = JSON3.read.(readlines(joinpath(dir, "threaded", "stages.jsonl")))
        @test all(r -> r.seconds >= 0 && haskey(r, :parent_stage), stages)
        config = JSON3.read(read(joinpath(dir, "threaded", "config.json"), String))
        @test config.julia_threads == Threads.nthreads(:default)
        @test config.options.scf.parallel_tasks == 0
        @test isfile(joinpath(dir, "threaded", "cross-state.jls"))
        checkpoint = load_checkpoint(joinpath(dir, "threaded", "checkpoint.jls"))
        @test all(
            i -> checkpoint.orbitals[i].c == threaded.state.orbitals[i].c,
            eachindex(checkpoint.orbitals),
        )
        println(
            "Parallel SCF comparison: energy_max_difference_Ha=",
            maximum(abs.([r.total for r in threaded.history] .- [r.total for r in serial.history])),
            ", final_audit_residual_Ha=",
            threaded.audit.max_residual,
            ", physical_audit_passed=",
            threaded.audit.passed,
        )
    end
end
