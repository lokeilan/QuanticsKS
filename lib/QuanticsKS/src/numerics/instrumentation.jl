# Recorder tracks stages, samples, and timings; cross.jl performs TCI fits.

using JSON3, Dates, Printf
include("cross_state.jl")

mutable struct Recorder
    options::NamedTuple
    dir::Union{Nothing,String}
    scf::Int
    inner::Int
    stage::String
    calls::Int
    samples::Int
    tci_time::Float64
    stage_io::IO
    tci_io::IO
    cross_states::Dict{CrossKey,CrossState}
    cross_order::Vector{CrossKey}
end
function Recorder(dir; options)
    dir === nothing && return Recorder(options, nothing, 0, 0, "init", 0, 0, 0.0, devnull, devnull,
                                      Dict{CrossKey,CrossState}(), CrossKey[])
    mkpath(dir)
    Recorder(
        options,
        dir,
        0,
        0,
        "init",
        0,
        0,
        0.0,
        open(joinpath(dir, "stages.jsonl"), "w"),
        open(joinpath(dir, "tci_calls.jsonl"), "w"),
        Dict{CrossKey,CrossState}(),
        CrossKey[],
    )
end
function memory_info()
    if Sys.iswindows()
        v = zeros(UInt64, 10)
        v[1] = 80
        p = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
        ok = ccall(
            (:GetProcessMemoryInfo, "psapi"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt64}, UInt32),
            p,
            v,
            80,
        )
        ok != 0 && return (rss_bytes = v[3], peak_rss_bytes = v[2], private_bytes = v[10])
    end
    (rss_bytes = 0, peak_rss_bytes = Sys.maxrss(), private_bytes = 0)
end
function event!(io, row)
    JSON3.write(io, row)
    write(io, '\n')
    flush(io)
end
function timed(f, r::Recorder, id::String; item = "")
    prev = r.stage
    r.stage = id
    c = r.calls
    s = r.samples
    t = r.tci_time
    try
        z = @timed with_svd_context(
            f;
            scf = r.scf,
            inner = r.inner,
            stage = id,
            diagnostic_dir = r.dir === nothing ? nothing : joinpath(r.dir, "numerics-failures"),
        )
        event!(
            r.stage_io,
            merge(
                (
                    scf = r.scf,
                    inner = r.inner,
                    stage = id,
                    parent_stage = prev,
                    item = item,
                    seconds = z.time,
                    gc_seconds = z.gctime,
                    allocated_bytes = z.bytes,
                    tci_calls = r.calls - c,
                    tci_samples = r.samples - s,
                    tci_seconds = r.tci_time - t,
                    compile_seconds = hasproperty(z, :compile_time) ? z.compile_time : 0.0,
                ),
                memory_info(),
            ),
        )
        return z.value
    finally
        r.stage = prev
    end
end
function Base.close(r::Recorder)
    r.dir === nothing && return nothing
    try
        isempty(r.cross_states) || save_cross_state(r, joinpath(r.dir, "cross-state.jls"))
    finally
        close(r.stage_io)
        close(r.tci_io)
    end
end
debug_snapshot(r::Recorder, name, data) =
    r.dir === nothing ? nothing : serialize(joinpath(r.dir, name), data)

evaluate(x, r::Recorder) = Evaluator(x; cachebits = r.options.environment_cache_bits)
block_oracle(xs, r::Recorder) = BlockOracle(
    xs;
    cap = r.options.orbital_cache_capacity,
    cachebits = r.options.environment_cache_bits,
    frame_capacity = get(r.options, :slice_cache_capacity, 2048),
)

include("cross.jl")
