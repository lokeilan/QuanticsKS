# Cyclic tridiagonal shift-invert and eigenpair refinement; recheck residuals with the original H.

# Included in H100Reference. Full-cell FD2 + Pulay/DIIS, with the linked
# reference's complex pure state at a spectrally degenerate Fermi pair.
# There is no spatial symmetry projection, averaging or cell reduction.
struct CyclicTridiagonalFactor
    pivots::Vector{Float64}
    lower::Vector{Float64}
    z::Vector{Float64}
    tail::Float64
    denominator::Float64
end
function solve_open!(y, pivots, lower, x)
    copyto!(y, x)
    @inbounds for i = 2:length(y)
        y[i] -= lower[i - 1] * y[i - 1]
    end
    y[end] /= pivots[end]
    @inbounds for i = (length(y) - 1):-1:1
        y[i] = y[i] / pivots[i] - lower[i] * y[i + 1]
    end
    return y
end
function CyclicTridiagonalFactor(potential, dx, shift)
    n = length(potential)
    hopping = -0.5 / dx^2
    diagonal = inv(dx^2) .+ potential .- shift
    gamma = -diagonal[1]
    tail = hopping / gamma
    diagonal[1] -= gamma
    diagonal[end] -= hopping * tail
    lower = zeros(n - 1)
    @inbounds for i = 2:n
        lower[i - 1] = hopping / diagonal[i - 1]
        diagonal[i] -= lower[i - 1] * hopping
    end
    minimum(diagonal) > 0 || error("Shifted open tridiagonal factor is not positive definite")
    u = zeros(n)
    u[1] = gamma
    u[end] = hopping
    z = similar(u)
    solve_open!(z, diagonal, lower, u)
    denominator = 1 + z[1] + tail * z[end]
    abs(denominator) > eps() || error("Cyclic correction is numerically singular")
    CyclicTridiagonalFactor(diagonal, lower, z, tail, denominator)
end
function solve_cyclic!(y, F::CyclicTridiagonalFactor, x)
    solve_open!(y, F.pivots, F.lower, x)
    correction = (y[1] + F.tail * y[end]) / F.denominator
    @. y -= correction * F.z
    return y
end

struct CyclicInverse <: AbstractMatrix{Float64}
    factor::CyclicTridiagonalFactor
    n::Int
    counter::Counter
    hamiltonian::Hamiltonian
    shift::Float64
    residual::Vector{Float64}
    correction::Vector{Float64}
end
Base.size(a::CyclicInverse) = (a.n, a.n)
LinearAlgebra.issymmetric(::CyclicInverse) = true
LinearAlgebra.ishermitian(::CyclicInverse) = true
function LinearAlgebra.mul!(y::AbstractVector, a::CyclicInverse, x::AbstractVector)
    t = time_ns()
    solve_cyclic!(y, a.factor, x)
    # Iterative refinement uses the declared FD2 stencil, rather than relying
    # solely on the accuracy of a factorization of large diagonal entries.
    for _ = 1:2
        mul!(a.residual, a.hamiltonian, y)
        @. a.residual = x - a.residual + a.shift * y
        solve_cyclic!(a.correction, a.factor, a.residual)
        y .+= a.correction
    end
    a.counter.calls += 3
    a.counter.columns += 1
    a.counter.seconds += (time_ns() - t) * 1e-9
    return y
end

function cyclic_matrix(potential, dx)
    n = length(potential)
    hopping = -0.5 / dx^2
    A = spdiagm(-1 => fill(hopping, n - 1), 0 => inv(dx^2) .+ potential, 1 => fill(hopping, n - 1))
    A[1, n] = hopping
    A[n, 1] = hopping
    return A
end

function refine_cyclic_eigenvectors!(vectors, inverse, h)
    polished = similar(vectors)
    for b in axes(vectors, 2)
        mul!(view(polished, :, b), inverse, view(vectors, :, b))
    end
    rdiv!(polished, cholesky(Symmetric(polished'polished)).U)
    hpolished = similar(polished)
    mul!(hpolished, h, polished)
    small = eigen(Symmetric(polished'hpolished))
    mul!(vectors, polished, small.vectors)
    return small.values
end

function maximum_cyclic_eigen_residual(h, vectors, values)
    work = zeros(size(vectors, 1))
    largest = 0.0
    for b in axes(vectors, 2)
        vector = view(vectors, :, b)
        mul!(work, h, vector)
        @. work -= values[b] * vector
        largest = max(largest, norm(work))
    end
    return largest
end

function cyclic_eigenpairs(
    h,
    k,
    r,
    it,
    seed,
    tol,
    maxiter;
    shift_margin = 0.05,
    target_tolerance = 1e-8,
)
    A = stage(r, it, "assemble_full_cyclic_tridiagonal") do
        cyclic_matrix(h.potential, h.dx)
    end
    # T is positive semidefinite. This shift is strictly below the spectrum;
    # no estimate of an unknown eigenvalue or spatial symmetry is required.
    shift = minimum(h.potential) - shift_margin
    F = stage(r, it, "shifted_cyclic_factorization") do
        CyclicTridiagonalFactor(h.potential, h.dx, shift)
    end
    inverse =
        CyclicInverse(F, size(A, 1), Counter(), h, shift, zeros(size(A, 1)), zeros(size(A, 1)))
    solved = stage(r, it, "full_cell_shiftinvert_lanczos") do
        eigs(
            inverse;
            nev = k,
            ncv = min(size(A, 1), max(2k + 12, 32)),
            which = :LM,
            tol = tol,
            maxiter = maxiter,
            v0 = randn(MersenneTwister(seed + it), size(A, 1)),
        )
    end
    mu, vectors, nconv, niter, nmult, resid = solved
    nconv == k || error("Only $nconv of $k full-cell eigenpairs converged")
    values = shift .+ inv.(mu)
    order = sortperm(values)
    values = values[order]
    vectors = vectors[:, order]
    true_res = stage(r, it, "independent_real_eigen_residual") do
        maximum_cyclic_eigen_residual(h, vectors, values)
    end
    # A fine-grid stencil magnifies tiny high-frequency roundoff components in
    # Ritz vectors. Standard inverse iteration removes them without changing
    # the Hamiltonian, occupations, or the independent acceptance tolerance.
    for pass = 1:3
        true_res <= target_tolerance && break
        values = stage(r, it, "cyclic_eigenvector_inverse_refinement_" * string(pass)) do
            refine_cyclic_eigenvectors!(vectors, inverse, h)
        end
        true_res = stage(r, it, "refined_real_eigen_residual_" * string(pass)) do
            maximum_cyclic_eigen_residual(h, vectors, values)
        end
    end
    open(joinpath(r.directory, "eigensolver_operations.csv"), "a") do io
        println(
            io,
            join(
                (
                    it,
                    nconv,
                    niter,
                    nmult,
                    inverse.counter.calls,
                    inverse.counter.seconds,
                    true_res,
                    shift,
                ),
                ',',
            ),
        )
    end
    return (; values, vectors, niter, nmult, true_res, matrix = A)
end
