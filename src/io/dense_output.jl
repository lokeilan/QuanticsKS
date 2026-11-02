# Dense timing, process memory, and array inventory; retained bytes and peak RSS are separate.

write_toml(path, data) = open(io -> TOML.print(io, data; sorted = true), path, "w")
function memory_plan(c::Config)
    n, k, no = 2^c.R, c.natoms ÷ 2 + c.extra_states, c.natoms ÷ 2
    return Dict(
        "R" => c.R,
        "grid_points" => n,
        "occupied_orbitals" => no,
        "solver_states" => k,
        "float64_bytes" => 8,
        "occupied_samples_bytes" => (c.eigensolver == "cyclic" ? 16 : 8) * n * no,
        "solver_block_bytes" => 8 * n * k,
        "hypothetical_full_matrix_bytes" => 8 * n * n,
        "full_matrix_allocated" => false,
        "note" => "Array sizes only. This is not a peak-RSS estimate. See arrays.csv after allocation.",
    )
end

"""OS process memory, including native BLAS/FFTW allocations. Bytes, not allocation totals."""
function memory_sample()
    rss = Ref{Csize_t}(0)
    rc = ccall(:uv_resident_set_memory, Cint, (Ref{Csize_t},), rss)
    return (rss = rc == 0 ? Int(rss[]) : -1, peak = Int(Sys.maxrss()))
end

mutable struct Recorder
    directory::String
    stages::IOStream
    events::IOStream
    scf::IOStream
    origin::UInt64
    sequence::Int
    stage_seconds::Float64
end

function Recorder(dir)
    stages = open(joinpath(dir, "stages.csv"), "w")
    events = open(joinpath(dir, "events.csv"), "w")
    scf = open(joinpath(dir, "scf.csv"), "w")
    println(
        stages,
        "sequence,scf,operation,wall_seconds,gc_seconds,julia_allocated_bytes,compile_seconds,recompile_seconds,rss_before_bytes,rss_after_bytes,process_peak_rss_bytes",
    )
    println(events, "sequence,scf,operation,event,elapsed_seconds,process_peak_rss_bytes")
    println(
        scf,
        "iteration,wall_seconds,eigen_iterations,energy_Ha,energy_change_Ha,density_relative_L2,nonlinear_residual_F_Ha,nonlinear_residual_RMS_Ha,nonlinear_residual_max_column_Ha,gram_F,charge_error,input_eigen_residual_max_Ha,fermi_gap_Ha,extra_block_end_gap_Ha,accepted",
    )
    return Recorder(dir, stages, events, scf, time_ns(), 0, 0.0)
end

function stage(f, r::Recorder, iteration, operation)
    r.sequence += 1
    seq = r.sequence
    before = memory_sample()
    println(
        r.events,
        join((seq, iteration, operation, "begin", (time_ns() - r.origin) * 1e-9, before.peak), ','),
    )
    flush(r.events)
    try
        t = @timed f()
        after = memory_sample()
        println(
            r.stages,
            join(
                (
                    seq,
                    iteration,
                    operation,
                    t.time,
                    t.gctime,
                    t.bytes,
                    t.compile_time,
                    t.recompile_time,
                    before.rss,
                    after.rss,
                    after.peak,
                ),
                ',',
            ),
        )
        flush(r.stages)
        r.stage_seconds += t.time
        println(
            r.events,
            join(
                (seq, iteration, operation, "end", (time_ns() - r.origin) * 1e-9, after.peak),
                ',',
            ),
        )
        flush(r.events)
        return t.value
    catch
        println(
            r.events,
            join(
                (seq, iteration, operation, "failed", (time_ns() - r.origin) * 1e-9, Sys.maxrss()),
                ',',
            ),
        )
        flush(r.events)
        rethrow()
    end
end

function close_recorder(r)
    close(r.stages)
    close(r.events)
    close(r.scf)
end
function inventory(path, roots)
    seen = IdDict{Any,Bool}()
    rows = Tuple[]
    function visit(name, a)
        a === nothing && return
        a isa Union{Number,Symbol,AbstractString,Function,Type,Module,Ptr,IO} && return
        if a isa SparseMatrixCSC
            visit(name * ".colptr", a.colptr)
            visit(name * ".rowval", a.rowval)
            visit(name * ".nzval", a.nzval)
            return
        end
        if a isa AbstractArray && isbitstype(eltype(a))
            p = parent(a)
            p !== a && return visit(name * ".parent", p)
            haskey(seen, a) && return
            seen[a] = true
            bytes = a isa BitArray ? sizeof(a.chunks) : sizeof(eltype(a)) * length(a)
            push!(rows, (name, string(eltype(a)), join(size(a), 'x'), bytes))
            return
        end
        # Plan handles own native memory; traversing their pointers cannot count it.
        a isa FFTW.FFTWPlan && return
        haskey(seen, a) && return
        seen[a] = true
        if a isa NamedTuple
            for k in keys(a)
                visit(name * "." * string(k), getfield(a, k))
            end
        elseif a isa Tuple
            for (i, v) in enumerate(a)
                visit(name * "." * string(i), v)
            end
        elseif parentmodule(typeof(a)) in (H100Reference, LinearAlgebra)
            for k in fieldnames(typeof(a))
                visit(name * "." * string(k), getfield(a, k))
            end
        end
    end
    visit("state", roots)
    open(path, "w") do io
        println(io, "name,element_type,dimensions,payload_bytes")
        for row in rows
            println(io, join(row, ','))
        end
    end
    return sum(row[4] for row in rows; init = 0)
end
