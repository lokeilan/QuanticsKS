# Fixed-H block eigensolver using orbitals, preconditioned residuals, and prior directions.

"""Full-grid blocks for a locally optimal [X, preconditioned residual, P] solve."""
struct BlockWorkspace
    X::Matrix{Float64}
    AX::Matrix{Float64}
    residual::Matrix{Float64}
    W::Matrix{Float64}
    P::Matrix{Float64}
    Z::Matrix{Float64}
    AZ::Matrix{Float64}
    nextX::Matrix{Float64}
    nextAX::Matrix{Float64}
end
function BlockWorkspace(X)
    n, k = size(X)
    BlockWorkspace(
        X,
        zeros(n, k),
        zeros(n, k),
        zeros(n, k),
        zeros(n, k),
        zeros(n, 3k),
        zeros(n, 3k),
        zeros(n, k),
        zeros(n, k),
    )
end

"""Append a numerically independent orthonormal block. Rank truncation acts on
search directions only, never on physical coordinates or occupied orbitals.
"""
function append_basis!(Z, q, W; rank_tolerance = 1e-10)
    n, m = size(W)
    # Normalize columns before rank detection; residual amplitudes differ greatly.
    for j = 1:m
        z = norm(view(W, :, j))
        z > 0 && (view(W, :, j) ./= z)
    end
    if q > 0
        old = view(Z, :, 1:q)
        for _ = 1:2
            overlap = old'W
            mul!(W, old, overlap, -1.0, 1.0)
        end
    end
    f = qr!(W, ColumnNorm())
    rank = count(>(rank_tolerance), abs.(diag(f.R)))
    rank == 0 && return q
    new = view(Z, :, (q + 1):(q + rank))
    fill!(new, 0.0)
    for j = 1:rank
        new[j, j] = 1.0
    end
    lmul!(f.Q, new)
    if q > 0
        old = view(Z, :, 1:q)
        for _ = 1:2
            overlap = old'new
            mul!(new, old, overlap, -1.0, 1.0)
        end
    end
    # The retained QR columns are independent. Reorthogonalization removes the
    # roundoff component along the existing basis before a small Gram correction.
    factor = cholesky(Symmetric(new'new))
    rdiv!(new, factor.U)
    return q + rank
end

function block_eigensolve!(s::BlockWorkspace, h, p; tol, maxiter, trace_io = nothing)
    X = s.X
    n, k = size(X)
    fill!(s.P, 0.0)
    trace = NamedTuple[]
    t0 = time_ns()
    copyto!(s.W, X)
    rank = append_basis!(s.Z, 0, s.W)
    rank == k || error("Initial orbital block is rank deficient")
    copyto!(X, view(s.Z, :, 1:k))
    mul!(s.AX, h, X)
    e = eigen(Symmetric(X's.AX))
    mul!(s.nextX, X, e.vectors)
    mul!(s.nextAX, s.AX, e.vectors)
    copyto!(X, s.nextX)
    copyto!(s.AX, s.nextAX)
    values = e.values
    init_seconds = (time_ns() - t0) * 1e-9
    if trace_io !== nothing
        println(
            trace_io,
            "iteration,max_residual_Ha,residual_seconds,preconditioner_seconds,basis_seconds,H_times_basis_seconds,rayleigh_ritz_seconds,update_seconds,wall_seconds,trial_dimension",
        )
        flush(trace_io)
    end
    for it = 1:maxiter
        start = time_ns()
        t = time_ns()
        # Re-evaluate the actual operator residual; propagated AX products can
        # accumulate roundoff on fine finite-difference grids.
        mul!(s.AX, h, X)
        @inbounds for j = 1:k, i = 1:n
            s.residual[i, j] = s.AX[i, j] - values[j] * X[i, j]
        end
        maxres = maximum(norm(view(s.residual, :, j)) for j = 1:k)
        residual_seconds = (time_ns() - t) * 1e-9
        pre = basis = action = rr = update = 0.0
        q = k
        done = maxres <= tol
        if !done && it < maxiter
            t = time_ns()
            ldiv!(s.W, p, s.residual)
            pre = (time_ns() - t) * 1e-9
            t = time_ns()
            copyto!(view(s.Z, :, 1:k), X)
            q = append_basis!(s.Z, k, s.W)
            copyto!(s.W, s.P)
            q = append_basis!(s.Z, q, s.W)
            q > k || error("Eigensolver search space stalled before residual convergence")
            basis = (time_ns() - t) * 1e-9
            Z = view(s.Z, :, 1:q)
            AZ = view(s.AZ, :, 1:q)
            t = time_ns()
            mul!(AZ, h, Z)
            action = (time_ns() - t) * 1e-9
            t = time_ns()
            # Solve the small generalized Ritz problem using the measured Gram
            # matrix, preserving full-cell vectors and all k requested states.
            small = eigen(Symmetric(Z'AZ), Symmetric(Z'Z))
            C = view(small.vectors, :, 1:k)
            rr = (time_ns() - t) * 1e-9
            t = time_ns()
            mul!(s.P, view(Z, :, (k + 1):q), view(C, (k + 1):q, :))
            mul!(s.nextX, Z, C)
            mul!(s.nextAX, AZ, C)
            copyto!(X, s.nextX)
            copyto!(s.AX, s.nextAX)
            values = small.values[1:k]
            update = (time_ns() - t) * 1e-9
        end
        row = (
            iteration = it,
            max_residual = maxres,
            residual_seconds = residual_seconds,
            preconditioner_seconds = pre,
            basis_seconds = basis,
            action_seconds = action,
            rayleigh_ritz_seconds = rr,
            update_seconds = update,
            wall_seconds = (time_ns() - start) * 1e-9,
            trial_dimension = q,
        )
        push!(trace, row)
        if trace_io !== nothing
            println(trace_io, join(Tuple(row), ','))
            flush(trace_io)
        end
        if done || it == maxiter
            return (
                λ = values,
                X = X,
                converged = done,
                iterations = it,
                trace = trace,
                initialization_seconds = init_seconds,
            )
        end
    end
    error("Unreachable")
end

"""List unique Julia-owned numeric array buffers, including block-solver workspaces.
Views/adjoints refer to their owning arrays. Native FFT/BLAS memory is in RSS only.
"""
