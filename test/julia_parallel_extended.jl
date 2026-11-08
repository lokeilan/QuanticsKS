using Test, QuanticsKS, LinearAlgebra, Random, JSON3, Serialization, TOML
const K = QuanticsKS
const Q = K.Q
BLAS.set_num_threads(1)
function dense(x)
    evaluator = Q.Evaluator(x; cachebits = 0)
    [evaluator(j) for j = 0:(2^length(x) - 1)]
end

@testset "Recorded fits preserve ordered RNG, pivots, caches and receipts" begin
    mktempdir() do dir
        function trial(tasks, name)
            Random.seed!(20261104)
            rec = K.recorder(
                joinpath(dir, name),
                TCIOptions(max_iterations = 8, environment_cache_bits = 0);
                parallel_tasks = tasks,
            )
            evaluations = Ref(0)
            field = Q.RecordedSamples(Q.sampled(j -> begin
                evaluations[] += 1
                2 + cos(2pi * j / 64)
            end, 6), rec.options)
            shared = Q.synchronized(field)
            jobs = Set{Task}()
            mutex = ReentrantLock()
            trains = Q.QTT[]
            for pass = 1:2
                trains = Q.recorded_map(
                    Q.QTT,
                    8,
                    (; parallel_tasks = tasks),
                    rec;
                    group_size = 2,
                ) do i, localrec
                    lock(mutex) do
                        push!(jobs, current_task())
                    end
                    sleep(0.002)
                    Q.cross(
                        localrec,
                        6,
                        Q.pointwise(v -> v + 0.1cis(i), shared);
                        tol = 1e-11,
                        cap = 16,
                        label = "fit_$i",
                        shared_pivots = "group_$(cld(i, 2))",
                    )
                end
            end
            state = Q.cross_state_data(rec)
            tail = rand(5)
            count = rec.calls
            close(rec)
            rows = JSON3.read.(readlines(joinpath(dir, name, "tci_calls.jsonl")))
            @test [r.call for r in rows] == collect(1:count)
            @test [r.label for r in rows] == repeat(["fit_$i" for i = 1:8], 2)
            @test rows[2].pivot_source == "orbital_group"
            @test all(r -> r.pivot_source == "previous_fit", rows[9:end])
            @test evaluations[] == field.evaluations <= 64
            @test all(
                i ->
                    norm(dense(trains[i]) - [2 + cos(2pi * j / 64) + 0.1cis(i) for j = 0:63]) < 1e-9,
                1:8,
            )
            @test isfile(joinpath(dir, name, "cross-state.jls"))
            (; trains, state, tail, count, jobs)
        end
        serial = trial(1, "serial")
        threaded = trial(0, "threaded")
        @test serial.tail == threaded.tail
        @test serial.state == threaded.state
        @test serial.count == threaded.count
        @test all(i -> dense(serial.trains[i]) == dense(threaded.trains[i]), 1:8)
        if Threads.nthreads(:default) > 1
            @test length(threaded.jobs) > 1
        end

        rec = K.recorder(joinpath(dir, "failure"), TCIOptions())
        finished = Threads.Atomic{Int}(0)
        started = Threads.Atomic{Int}(0)
        workers = min(4, Threads.nthreads(:default))
        @test_throws ErrorException Q.recorded_map(
            Q.QTT,
            4,
            (; parallel_tasks = 0),
            rec,
        ) do i, localrec
            Threads.atomic_add!(started, 1)
            while started[] < workers
                yield()
            end
            i == 1 && error("injected fit failure")
            x = Q.cross(
                localrec,
                4,
                j -> 2 + cos(2pi * j / 16 + i);
                tol = 1e-10,
                cap = 8,
                label = "survivor_$i",
            )
            Threads.atomic_add!(finished, 1)
            x
        end
        @test finished[] == workers - 1
        @test rec.calls == workers - 1
        close(rec)
        stages = JSON3.read.(readlines(joinpath(dir, "failure", "stages.jsonl")))
        @test any(r -> get(r, :event, "") == "parallel_group_failure", stages)
    end
end

@testset "Nested budgets and cancellation-safe matrices" begin
    @test_throws ArgumentError K.recorder(nothing, TCIOptions(); parallel_tasks = 1.5)
    @test_throws ArgumentError Q.parallel_workers((; parallel_tasks = 1.5), 4)
    nested = Q.independent_map(
        i -> Q.parallel_workers((; parallel_tasks = 0), 20),
        Int,
        8,
        (; parallel_tasks = 0),
    )
    @test all(==(1), nested)
    Q.foreach_independent(1, (; parallel_tasks = 1)) do _
        @test Q.parallel_workers((; parallel_tasks = 0), 20) == 1
    end
    @test !haskey(task_local_storage(), Q.PARALLEL_BUDGET_KEY)
    p = Problem(PeriodicGrid1D(5, 8), LocalPotential1D(0.2), [2.0, 2.0])
    state = state_from_functions(p, [x -> 1, x -> cis(2pi * x / 8)])
    xs = state.orbitals
    v = Q.constant_tt(5, 0.2)
    H = Q.Hamiltonian(v, Q.diagonal(v), Q.shift_mpo(5, 1), Q.shift_mpo(5, -1), 0.25)
    @test Q.gram(xs; cfg = (; parallel_tasks = 0)) == Q.gram(xs)
    @test Q.h_gram(xs, H; cfg = (; parallel_tasks = 0)) ≈ Q.h_gram(xs, H) atol = 1e-12
    @test Q.h_gram(xs, H; cfg = (; parallel_tasks = 0))[1, 1] ≈ 0.2 atol = 1e-12
    left, right = Q.bits.(0:3, 2), Q.bits.(0:1, 1)
    serial = Q.BlockOracle(xs; cachebits = 0)
    threaded = Q.BlockOracle(xs; cachebits = 0, parallel_tasks = 0)
    @test serial(left, right, Val(2)) == threaded(left, right, Val(2))
    serial_table = Q.XCTable(npoints = 16)
    threaded_table = Q.XCTable(npoints = 16, parallel_tasks = 0)
    @test serial_table.energy_density == threaded_table.energy_density
    @test serial_table.log_density_derivative == threaded_table.log_density_derivative
end

@testset "Fixed-H methods, strict output Ritz, refinement and propagation" begin
    p = Problem(
        PeriodicGrid1D(5, 8),
        LocalPotential1D(x -> 0.2cos(2pi * x / 8); id = "full-parallel"),
        [2.0, 2.0],
    )
    a = Accuracy(
        field_tolerance = 1e-11,
        work_tolerance = 1e-9,
        storage_tolerance = 1e-11,
        orbital_cap = 32,
        field_cap = 32,
        work_cap = 64,
    )
    Random.seed!(42)
    state = state_from_functions(p, [x -> 1, x -> cis(2pi * x / 8)]; accuracy = a)
    potential = [0.2cos(2pi * j / 32) for j = 0:31]
    for (method, basis, projection, diagnostics) in (
        (:block_green, :direct, :streaming_grid, :streaming_grid),
        (:block_residual, :direct, :qtt_gradient, :qtt),
        (:block_residual, :projected, :streaming_grid, :qtt),
        (:lrrap, :direct, :qtt_gradient, :qtt),
    )
        options = FixedSpectrumOptions(;
            states = 3,
            occupied_states = 2,
            virtual_wavevectors = [-1],
            virtual_envelope_period = 8.0,
            min_iterations = 2,
            max_iterations = 2,
            method,
            search_basis = basis,
            projection,
            diagnostics,
            grid_block_size = 9,
        )
        serial = fixed_spectrum(
            state,
            potential;
            spectrum = options,
            accuracy = a,
            solver = SCFOptions(parallel_tasks = 1),
        )
        threaded = fixed_spectrum(
            state,
            potential;
            spectrum = options,
            accuracy = a,
            solver = SCFOptions(parallel_tasks = 0),
        )
        @test threaded.eigenvalues ≈ serial.eigenvalues atol = 1e-9
        @test threaded.converged == serial.converged
        @test maximum(
            norm(dense(x) - dense(y)) for (x, y) in zip(serial.orbitals, threaded.orbitals)
        ) < 1e-8
    end
    results = [
        ground_state(
            p;
            initial = state,
            accuracy = a,
            scf = SCFOptions(parallel_tasks = tasks, criterion = :strict, max_iterations = 2),
            tci = TCIOptions(orbital_pivot_group_size = 1),
        ) for tasks in (1, 0)
    ]
    @test [r.total for r in results[1].history] ≈ [r.total for r in results[2].history] atol = 1e-10
    @test last(results[2].history).final_output_ritz
    @test results[1].audit.orbital_residuals ≈ results[2].audit.orbital_residuals atol = 1e-9
    target = Problem(PeriodicGrid1D(6, 8), p.model, p.occupations)
    Random.seed!(31)
    first = refine_state(state, target; accuracy = a, parallel_tasks = 1)
    Random.seed!(31)
    second = refine_state(state, target; accuracy = a, parallel_tasks = 0)
    @test all(i -> dense(first.orbitals[i]) ≈ dense(second.orbitals[i]), 1:2)
    runs = [
        propagate(
            state;
            accuracy = a,
            time = TimeOptions(
                parallel_tasks = tasks,
                dt = 0.001,
                steps = 2,
                output_cap = 32,
                phase_cap = 32,
                kinetic_cap = 64,
                norm_drift_limit = 1e-7,
                gram_drift_limit = 1e-7,
            ),
        ) for tasks in (1, 0)
    ]
    @test maximum(
        norm(dense(x) - dense(y)) for (x, y) in zip(runs[1].state.orbitals, runs[2].state.orbitals)
    ) < 1e-10
    @test last(runs[1].observations).energy.total ≈ last(runs[2].observations).energy.total atol =
        1e-10
    mktempdir() do dir
        run = joinpath(dir, "time")
        propagate(
            state;
            accuracy = a,
            output = run,
            time = TimeOptions(
                parallel_tasks = 0,
                dt = 0.001,
                steps = 2,
                snapshot_every = 1,
                output_cap = 32,
                phase_cap = 32,
                kinetic_cap = 64,
                norm_drift_limit = 1e-7,
                gram_drift_limit = 1e-7,
            ),
        )
        Random.seed!(39)
        output = postprocess(run; accuracy = a, samples = 16, parallel_tasks = 1, model = p.model)
        serial = JSON3.read(read(joinpath(output, "trajectory.json"), String))
        Random.seed!(39)
        postprocess(run; accuracy = a, samples = 16, parallel_tasks = 0, model = p.model)
        threaded = JSON3.read(read(joinpath(output, "trajectory.json"), String))
        @test [row.energy.total for row in serial] ≈ [row.energy.total for row in threaded] atol = 1e-10
        @test [row.time for row in serial] == [row.time for row in threaded]
        @test_throws ArgumentError postprocess(run; drive = CosinePulse(duration = -1.0))
    end
end
