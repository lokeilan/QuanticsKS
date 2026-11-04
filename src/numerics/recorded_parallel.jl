# Fit batches own their recorder and RNG; only ordered receipts reach the parent.

function fork_recorder(rec::Recorder)
    Recorder(
        rec.options,
        rec.dir,
        rec.scf,
        rec.inner,
        rec.stage,
        0,
        0,
        0.0,
        rec.stage_io === devnull ? devnull : IOBuffer(),
        rec.tci_io === devnull ? devnull : IOBuffer(),
        copy(rec.cross_states),
        copy(rec.cross_order),
        Pair{CrossKey,CrossState}[],
    )
end

function merge_recorder!(parent, child; sampling_seed = nothing, group = nothing)
    offset = parent.calls
    lines =
        child.tci_io === devnull ? String[] :
        split(String(take!(child.tci_io)), '\n'; keepempty = false)
    for line in lines
        row = Dict{String,Any}(String(k) => v for (k, v) in pairs(JSON3.read(line)))
        haskey(row, "call") && (row["call"] += offset)
        sampling_seed === nothing || get!(row, "sampling_seed", string(sampling_seed))
        group === nothing || get!(row, "parallel_group", group)
        event!(parent.tci_io, row)
    end
    if child.stage_io !== devnull
        write(parent.stage_io, take!(child.stage_io))
        flush(parent.stage_io)
    end
    parent.calls += child.calls
    parent.samples += child.samples
    parent.tci_time += child.tci_time
    for (key, state) in child.cross_updates
        old = get(parent.cross_states, key, nothing)
        updated = CrossState(
            state.R,
            state.left,
            state.right,
            state.ranks,
            state.errors,
            old === nothing ? 1 : old.uses + 1,
        )
        parent.cross_states[key] = updated
        parent.cross_updates === nothing || push!(parent.cross_updates, key => updated)
        filter!(!=(key), parent.cross_order)
        push!(parent.cross_order, key)
        while length(parent.cross_order) > get(parent.options, :pivot_cache_capacity, 512)
            delete!(parent.cross_states, popfirst!(parent.cross_order))
        end
    end
    nothing
end

"Fit independent groups from one pivot snapshot and merge logs in input order."
function recorded_map(
    f,
    ::Type{T},
    n::Integer,
    cfg,
    rec::Recorder;
    group_size = 1,
    random = true,
) where {T}
    n >= 0 && group_size > 0 || throw(ArgumentError("Invalid recorded batch dimensions"))
    n == 0 && return T[]
    groups = cld(n, group_size)
    seeds = random ? rand(UInt64, groups) : fill(UInt64(0), groups)
    children = [fork_recorder(rec) for _ = 1:groups]
    result = Vector{T}(undef, n)
    failures = fill!(Vector{Any}(undef, groups), nothing)
    cancelled = Threads.Atomic{Bool}(false)
    foreach_independent(groups, cfg) do group
        cancelled[] && return
        child = children[group]
        first = (group - 1) * group_size + 1
        last = min(group * group_size, n)
        try
            run = () -> begin
                for i = first:last
                    result[i] = f(i, child)
                end
            end
            with_svd_context(;
                parallel_group = group,
                sampling_seed = random ? string(seeds[group]) : nothing,
            ) do
                random ? with_sampling_seed(run, seeds[group]) : run()
            end
        catch exception
            cancelled[] = true
            failures[group] = exception
            event!(
                child.stage_io,
                (;
                    scf = child.scf,
                    inner = child.inner,
                    stage = child.stage,
                    event = "parallel_group_failure",
                    first_item = first,
                    last_item = last,
                    error = sprint(showerror, exception),
                    seconds = 0.0,
                    parent_stage = rec.stage,
                    tci_calls = 0,
                    tci_samples = 0,
                    tci_seconds = 0.0,
                ),
            )
        end
    end
    # Even a failed group retains its completed fits and diagnostic events.
    for (group, child) in enumerate(children)
        merge_recorder!(rec, child; sampling_seed = random ? seeds[group] : nothing, group)
    end
    failed = findfirst(!isnothing, failures)
    failed === nothing || throw(failures[failed])
    event!(
        rec.stage_io,
        (;
            scf = rec.scf,
            inner = rec.inner,
            stage = rec.stage,
            event = "parallel_batch",
            seconds = 0.0,
            parent_stage = rec.stage,
            items = n,
            groups,
            group_size,
            workers = parallel_workers(cfg, groups),
            tci_calls = 0,
            tci_samples = 0,
            tci_seconds = 0.0,
            sampling_rng = random ? "ordered_group_seeds" : "unused",
        ),
    )
    result
end
