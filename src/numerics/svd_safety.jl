# SVD checks, failure records, and retries; diagnostic context is local to the current task.

using Serialization, Dates, JSON3

const SVD_CONTEXT_KEY = :quanticsks_svd_context
const SVD_EVENT_ID = Threads.Atomic{Int}(0)

"Attach diagnostic data without copying tensors or touching the sampling RNG."
function with_svd_context(f; context...)
    tls = task_local_storage()
    previous = get(tls, SVD_CONTEXT_KEY, nothing)
    tls[SVD_CONTEXT_KEY] = merge(previous === nothing ? (;) : previous, (; context...))
    try
        f()
    finally
        previous === nothing ? delete!(tls, SVD_CONTEXT_KEY) : (tls[SVD_CONTEXT_KEY] = previous)
    end
end

function save_svd_event(A, reason; details = nothing)
    context = get(task_local_storage(), SVD_CONTEXT_KEY, (;))
    dir = get(context, :diagnostic_dir, nothing)
    dir === nothing &&
        (dir = get(ENV, "QUANTICSKS_SVD_DIAGNOSTICS", joinpath(pwd(), "numerics-failures")))
    mkpath(dir)
    id = Threads.atomic_add!(SVD_EVENT_ID, 1) + 1
    stem = joinpath(dir, "svd-$(getpid())-$(time_ns())-$id")
    data = (;
        schema_version = 1,
        recorded_utc = string(now(UTC)),
        reason,
        matrix = copy(A),
        context,
        details,
    )
    open(io -> serialize(io, data), stem * ".jls.tmp", "w")
    mv(stem * ".jls.tmp", stem * ".jls")
    scalar_context = Dict(
        string(k) => v for (k, v) in pairs(context) if
        v === nothing || v isa Union{AbstractString,Symbol,Number,Bool}
    )
    summary = Dict{String,Any}(
        "reason" => reason,
        "matrix_size" => size(A),
        "input_finite" => all(isfinite, A),
        "context" => scalar_context,
        "data_file" => stem * ".jls",
        "status" => "captured",
    )
    open(io -> JSON3.write(io, summary), stem * ".json", "w")
    stem, summary
end

function finish_svd_event(event, status; error = nothing)
    stem, summary = event
    summary["status"] = status
    error === nothing || (summary["error"] = error)
    open(io -> JSON3.write(io, summary), stem * ".json", "w")
    nothing
end

function require_finite(A, reason; details = nothing)
    all(isfinite, A) && return nothing
    event = save_svd_event(A, reason; details)
    finish_svd_event(event, "rejected_nonfinite_input")
    error("$reason; diagnostic data: $(first(event)).jls")
end

finite_svd(f) = all(isfinite, f.U) && all(isfinite, f.S) && all(isfinite, f.Vt)
retryable_svd_error(e) =
    e isa LinearAlgebra.LAPACKException ||
    (e isa ArgumentError && occursin("invalid argument #4 to LAPACK call", sprint(showerror, e)))

# factorize is an internal injection point for exercising backend failures.
# The public path always uses nonmutating LinearAlgebra.svd, preserving A for QR.
function checked_svd(A; full = false, factorize = svd)
    require_finite(A, "nonfinite_svd_input")
    event = nothing
    f = try
        factorize(A; full, alg = LinearAlgebra.DivideAndConquer())
    catch e
        retryable_svd_error(e) || rethrow()
        event = save_svd_event(
            A,
            "divide_and_conquer_exception";
            details = (; exception_type = string(typeof(e)), exception = sprint(showerror, e)),
        )
        nothing
    end
    if f !== nothing
        finite_svd(f) && return f
        event = save_svd_event(A, "nonfinite_svd_output"; details = (; U = f.U, S = f.S, Vt = f.Vt))
    end
    try
        retry = factorize(A; full, alg = LinearAlgebra.QRIteration())
        if !finite_svd(retry)
            failed = save_svd_event(
                A,
                "nonfinite_qr_output";
                details = (; U = retry.U, S = retry.S, Vt = retry.Vt),
            )
            finish_svd_event(failed, "failed")
            error("QR SVD returned nonfinite factors; diagnostic data: $(first(failed)).jls")
        end
        finish_svd_event(event, "recovered_with_qr")
        retry
    catch e
        finish_svd_event(event, "qr_failed"; error = sprint(showerror, e))
        rethrow()
    end
end

"TCI 0.9.19 SVD factorization, preserving its local tail criterion exactly."
function guarded_svd_factorize(A; tolerance, maxbonddim, leftorthogonal, normalizeerror = true)
    f = checked_svd(A)
    reltol, abstol = normalizeerror ? (tolerance, 0.0) : (1e-14, tolerance)
    err = [sum(f.S[(n + 1):end] .^ 2) for n = 1:length(f.S)]
    normalized_err = err ./ sum(f.S .^ 2)
    # The library keeps all available singular vectors at tolerance=0,
    # including the all-zero case. Keep its strict comparisons and cap.
    rank = min(
        something(findfirst(<(abstol^2), err), length(err)),
        something(findfirst(<(reltol^2), normalized_err), length(err)),
        maxbonddim,
    )
    left, right =
        leftorthogonal ? (f.U[:, 1:rank], Diagonal(f.S[1:rank]) * f.Vt[1:rank, :]) :
        (f.U[:, 1:rank] * Diagonal(f.S[1:rank]), f.Vt[1:rank, :])
    require_finite(left, "nonfinite_left_svd_factor"; details = (; input = A, S = f.S))
    require_finite(right, "nonfinite_right_svd_factor"; details = (; input = A, S = f.S))
    left, right, rank
end

"Validate bond shapes before a TensorTrain constructor can discard diagnostic context."
function check_compression_shapes(cores)
    for (k, g) in enumerate(cores)
        if any(iszero, size(g)) || (k > 1 && size(g, 1) != size(cores[k - 1], ndims(cores[k - 1])))
            with_svd_context(; site = k, direction = "input") do
                event = save_svd_event(g, "invalid_tensor_shape")
                finish_svd_event(event, "rejected_shape")
                throw(
                    DimensionMismatch("Invalid tensor shape; diagnostic data: $(first(event)).jls"),
                )
            end
        end
    end
    nothing
end

"Use library compression; replay with diagnostics only after a numerical failure."
function guarded_compress!(tt; tolerance = 1e-12, maxbonddim = typemax(Int), normalizeerror = true)
    check_compression_shapes(tt.sitetensors)
    for (k, g) in enumerate(tt.sitetensors)
        with_svd_context(; site = k, direction = "input") do
            require_finite(g, "nonfinite_compression_input")
        end
    end
    # TCI replaces cores without mutating the originals, so a shallow copy is enough.
    original_cores = copy(tt.sitetensors)
    try
        TCI.compress!(tt, :SVD; tolerance, maxbonddim, normalizeerror)
        all(core -> all(isfinite, core), tt.sitetensors) && return nothing
    catch error
        intermediate_nonfinite =
            error isa ArgumentError &&
            occursin("matrix contains Infs or NaNs", sprint(showerror, error))
        retryable_svd_error(error) || intermediate_nonfinite || rethrow()
    end
    tt.sitetensors .= original_cores
    # The pinned library has no factorization callback for checked QR recovery.
    for k = 1:(length(tt) - 1)
        with_svd_context(; site = k, direction = "forward") do
            sh = size(tt.sitetensors[k])
            A = reshape(tt.sitetensors[k], prod(sh[1:(end - 1)]), sh[end])
            left, right, rank = guarded_svd_factorize(
                A;
                tolerance = 0.0,
                maxbonddim = typemax(Int),
                leftorthogonal = true,
            )
            tt.sitetensors[k] = reshape(left, sh[1:(end - 1)]..., rank)
            sn = size(tt.sitetensors[k + 1])
            neighbor = reshape(tt.sitetensors[k + 1], sn[1], prod(sn[2:end]))
            next = right * neighbor
            require_finite(next, "nonfinite_forward_contraction"; details = (; right, neighbor))
            tt.sitetensors[k + 1] = reshape(next, rank, sn[2:end]...)
        end
    end
    for k = length(tt):-1:2
        with_svd_context(; site = k, direction = "backward") do
            sh = size(tt.sitetensors[k])
            A = reshape(tt.sitetensors[k], sh[1], prod(sh[2:end]))
            left, right, rank = guarded_svd_factorize(
                A;
                tolerance,
                maxbonddim,
                normalizeerror,
                leftorthogonal = false,
            )
            tt.sitetensors[k] = reshape(right, rank, sh[2:end]...)
            sp = size(tt.sitetensors[k - 1])
            neighbor = reshape(tt.sitetensors[k - 1], prod(sp[1:(end - 1)]), sp[end])
            next = neighbor * left
            require_finite(next, "nonfinite_backward_contraction"; details = (; neighbor, left))
            tt.sitetensors[k - 1] = reshape(next, sp[1:(end - 1)]..., rank)
        end
    end
    nothing
end
