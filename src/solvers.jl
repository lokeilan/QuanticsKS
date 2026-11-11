# Standard Block LOBPCG uses the same library iteration for both representations.

struct BlockHamiltonian{F}
    apply::F
end
function LinearAlgebra.mul!(result::AbstractMatrix, H::BlockHamiltonian, orbitals::AbstractMatrix)
    copyto!(result, H.apply(orbitals))
end

"An optional callback applies the inverse preconditioner to a whole block."
struct BlockPreconditioner{F}
    apply::F
end
function LinearAlgebra.ldiv!(preconditioner::BlockPreconditioner, residuals::AbstractMatrix)
    copyto!(residuals, preconditioner.apply(residuals))
end

"""
    block_lobpcg(H, orbitals; preconditioner=nothing, tolerance=1e-6,
                 max_iterations=100, gram_tolerance=1e-8, callback=identity)

Find the lowest Hermitian eigenpairs using LOBPCGEigensolver's block iteration.
`H` is a matrix or a block-action callback. A preconditioner callback applies
an inverse to a whole residual block; dense matrices and factorizations are
applied through a linear solve. Omission uses the identity.
The final residuals are recomputed from the returned orbitals independently
of the library's cached Hamiltonian actions. `H` must remain fixed throughout.
"""
function block_lobpcg(
    H,
    orbitals::AbstractMatrix;
    preconditioner = nothing,
    tolerance = 1e-6,
    max_iterations = 100,
    gram_tolerance = 1e-8,
    callback = identity,
)
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("Invalid residual tolerance"))
    isfinite(gram_tolerance) && gram_tolerance > 0 || throw(ArgumentError("Invalid Gram tolerance"))
    max_iterations >= 0 || throw(ArgumentError("Invalid iteration limit"))
    size(orbitals, 2) > 0 || throw(ArgumentError("An initial orbital block is required"))
    operator = H isa Union{AbstractMatrix,BlockHamiltonian} ? H : BlockHamiltonian(H)
    inverse =
        preconditioner === nothing ? I :
        preconditioner isa AbstractMatrix ? factorize(preconditioner) :
        preconditioner isa Union{Factorization,UniformScaling,BlockPreconditioner} ?
        preconditioner : BlockPreconditioner(preconditioner)
    started = time_ns()
    residual_history = Vector{Vector{Float64}}()
    residual_norms = fill(Inf, size(orbitals, 2))
    eigenvalues = zeros(size(orbitals, 2))
    locked = 0
    monitor =
        info -> begin
            active = (locked + 1):length(residual_norms)
            residual_norms[active] = info.resid_history[active, info.n_iter + 1]
            eigenvalues[active] = info.λs
            push!(residual_history, copy(residual_norms))
            locked = info.n_locked
            callback((;
                iteration = info.n_iter,
                eigenvalues = copy(eigenvalues),
                residual_norms = copy(residual_norms),
                locked,
            ))
        end
    result = LOBPCGEigensolver.lobpcg(
        operator,
        orbitals,
        I,
        inverse,
        tolerance,
        max_iterations;
        miniter = 0,
        callback = monitor,
    )
    stored_orbitals, residual_norms, gram_defect = lobpcg_audit(operator, result.X, result.λ)
    (;
        orbitals = stored_orbitals,
        eigenvalues = result.λ,
        residual_norms,
        gram_defect,
        converged = maximum(residual_norms) <= tolerance && gram_defect <= gram_tolerance,
        iterations = length(residual_history) - 1,
        residual_history = reduce(hcat, residual_history),
        elapsed_seconds = (time_ns() - started) / 1e9,
    )
end

function lobpcg_audit(H, orbitals::AbstractMatrix, energies)
    action = similar(orbitals)
    mul!(action, H, orbitals)
    residuals = action - orbitals * Diagonal(energies)
    norms = sqrt.(vec(sum(abs2, orbitals; dims = 1)))
    residual_norms = sqrt.(vec(sum(abs2, residuals; dims = 1))) ./ norms
    orbitals, residual_norms, norm(orbitals' * orbitals - I)
end

# Block algebra for the library LOBPCG loop. Scalar grid access is deliberately absent.

mutable struct QTTBlockWorkspace{C,R,H,P}
    accuracy::C
    recorder::R
    H::H
    potential::P
    actions::IdDict{Any,QTT}
    overlap_basis::Vector{QTT}
    overlap_positions::IdDict{Any,Int}
    overlap_metric::Matrix{ComplexF64}
end

"A block of deferred orbital expressions, or their symbolic Hamiltonian actions."
struct QTTBlock{V<:AbstractVector{QTTExpression},W} <: AbstractMatrix{ComplexF64}
    columns::V
    workspace::W
    action::Base.RefValue{Bool}
end

"Cache exact core overlaps within one iteration; transforms use only small matrices."
function block_core_gram(workspace, basis)
    missing = [psi for psi in basis if !haskey(workspace.overlap_positions, psi.c)]
    if !isempty(missing)
        old = workspace.overlap_basis
        cross_metric = gram(old, missing; cfg = workspace.accuracy)
        new_metric = gram(missing; cfg = workspace.accuracy)
        workspace.overlap_metric = [workspace.overlap_metric cross_metric; cross_metric' new_metric]
        for psi in missing
            push!(old, psi)
            workspace.overlap_positions[psi.c] = length(old)
        end
    end
    indices = [workspace.overlap_positions[psi.c] for psi in basis]
    workspace.overlap_metric[indices, indices]
end

function block_metric(x::QTTBlock)
    basis, coefficients = expression_components(x.columns)
    isempty(basis) && return zeros(ComplexF64, size(x, 2), size(x, 2))
    metric = block_core_gram(x.workspace, basis)
    result = coefficients' * metric * coefficients
    scales = real.(diag(abs.(coefficients)' * abs.(metric) * abs.(coefficients)))
    # A cancelled direction needs amplitudes. Do not build a global high-rank QR sum.
    if any(real.(diag(result)) .< sqrt(eps(Float64)) .* scales) &&
       any(e -> length(e.terms) > 1, x.columns)
        orbitals = reconstruct_orbitals!(x; label = "lobpcg_cancelled_direction")
        return block_core_gram(x.workspace, orbitals)
    end
    (result + result') / 2
end

QTTBlock(columns, workspace; action = false) = QTTBlock(columns, workspace, Ref(action))
Base.size(x::QTTBlock) = (1 << x.workspace.accuracy.R, length(x.columns))
Base.copy(x::QTTBlock) = QTTBlock(copy(x.columns), x.workspace; action = x.action[])
Base.view(x::QTTBlock, ::Colon, columns) = QTTBlock(view(x.columns, columns), x.workspace, x.action)
Base.getindex(x::QTTBlock, ::Colon, columns::AbstractVector) = copy(view(x, :, columns))
Base.getindex(x::QTTBlock, i::Int, j::Int) =
    error("Scalar grid traversal is not part of QTT block algebra; use evaluate explicitly")
function Base.similar(x::QTTBlock, ::Type{T}, dims::Dims) where {T}
    length(dims) == 2 && dims[1] == size(x, 1) && T == ComplexF64 || return Array{T}(undef, dims)
    zero_expression = QTTExpression(QTT[], ComplexF64[], x.workspace.accuracy.R)
    QTTBlock(fill(zero_expression, dims[2]), x.workspace)
end
Base.similar(x::QTTBlock) = similar(x, ComplexF64, size(x))
Base.similar(x::QTTBlock, ::Type{T}) where {T} = similar(x, T, size(x))
Base.similar(x::QTTBlock, dims::Int...) = similar(x, ComplexF64, dims)
Base.similar(x::QTTBlock, ::Type{T}, dims::Int...) where {T} = similar(x, T, dims)
Base.zero(x::QTTBlock) = similar(x)
function Base.copyto!(destination::QTTBlock, source::QTTBlock)
    size(destination) == size(source) || throw(DimensionMismatch("QTT block copy"))
    copyto!(destination.columns, source.columns)
    destination.action[] = source.action[]
    destination
end
Base.any(::typeof(isnan), x::QTTBlock) =
    any(x.columns) do expression
        any(isnan, expression.coefficients) ||
            any(t -> any(g -> any(isnan, g), t.c), expression.terms)
    end

"Materialize only when a consumer needs independent QTT cores."
function reconstruct_orbitals!(x::QTTBlock; label = "lobpcg_orbitals")
    x.action[] && error("Reconstruct the source orbitals of a Hamiltonian action")
    isempty(x.columns) && return QTT[]
    accuracy, rec = x.workspace.accuracy, x.workspace.recorder
    if all(e -> length(e.terms) == 1 && only(e.coefficients) == 1, x.columns)
        return [only(e.terms) for e in x.columns]
    end
    basis, coefficients = expression_components(x.columns)
    orbitals =
        isempty(basis) ? [constant_tt(accuracy.R, 0) for _ in x.columns] :
        combine_orbitals(
            basis,
            coefficients,
            accuracy,
            rec;
            tol = accuracy.orthogonalization_tolerance,
            cap = accuracy.orbital_cap,
            label,
        )
    x.columns .= [linear_expression((psi,), (1,)) for psi in orbitals]
    orbitals
end

function block_actions(x::QTTBlock)
    workspace = x.workspace
    accuracy = workspace.accuracy
    stored = [
        length(e.terms) == 1 && only(e.coefficients) == 1 ? only(e.terms) : nothing for
        e in x.columns
    ]
    missing = findall(
        i -> stored[i] === nothing || !haskey(workspace.actions, stored[i].c),
        eachindex(stored),
    )
    basis, coefficients = expression_components(x.columns)
    evaluators =
        any(isnothing, stored) ?
        orbital_evaluators(basis, length(missing), accuracy, workspace.recorder) : nothing
    computed = recorded_map(QTT, length(missing), accuracy, workspace.recorder) do k, rec
        i = missing[k]
        if stored[i] !== nothing
            return h_action(
                workspace.H,
                stored[i],
                accuracy,
                rec;
                work_tol = accuracy.work_tolerance,
                potential = workspace.potential,
                label = "lobpcg_$i",
            )
        end
        # Fit the complete action. Combining independently fitted H terms amplifies cancellation error.
        orbital = LinearSamples(orbital_evaluator(evaluators), coefficients[:, i])
        samples = hamiltonian_samples(workspace.H, orbital, workspace.potential, accuracy.R)
        cross(
            rec,
            accuracy.R,
            samples;
            tol = accuracy.work_tolerance,
            cap = accuracy.work_cap,
            label = "Hpsi_work_lobpcg_$i",
            initial_zero_fallback = () -> begin
                psi = reconstruct_expression(
                    x.columns[i];
                    tol = accuracy.orthogonalization_tolerance,
                    cap = accuracy.orbital_cap,
                )
                h_core_fallback(
                    workspace.H,
                    psi,
                    0.0;
                    tol = accuracy.work_tolerance,
                    cap = accuracy.work_cap,
                    rec,
                )
            end,
        )
    end
    actions = Vector{QTT}(undef, length(stored))
    for (i, action) in zip(missing, computed)
        actions[i] = action
        stored[i] === nothing || (workspace.actions[stored[i].c] = action)
    end
    for i in eachindex(stored)
        stored[i] === nothing || (actions[i] = workspace.actions[stored[i].c])
    end
    actions
end

function block_overlap(x::QTTBlock, y::QTTBlock)
    x.action[] && error("Hamiltonian action is only supported on the right of an overlap")
    if y.action[]
        actions = block_actions(y)
        basis, coefficients = expression_components(x.columns)
        return coefficients' * gram(basis, actions; cfg = x.workspace.accuracy)
    end
    x === y && return block_metric(x)
    nx = size(x, 2)
    basis, coefficients = expression_components(vcat(x.columns, y.columns))
    isempty(basis) && return zeros(ComplexF64, nx, size(y, 2))
    metric = block_core_gram(x.workspace, basis)
    coefficients[:, 1:nx]' * metric * coefficients[:, (nx + 1):end]
end
Base.:*(x::Adjoint{ComplexF64,<:QTTBlock}, y::QTTBlock) = block_overlap(parent(x), y)
function LOBPCGEigensolver.mul_hermi(x::Adjoint{ComplexF64,<:QTTBlock}, y::QTTBlock)
    metric = parent(x) === y && !y.action[] ? block_metric(y) : block_overlap(parent(x), y)
    Hermitian((metric + metric') / 2)
end
LOBPCGEigensolver.columnwise_norms(x::QTTBlock) = sqrt.(max.(0, real.(diag(block_metric(x)))))
LOBPCGEigensolver.columnwise_dots(x::QTTBlock, y::QTTBlock) = diag(block_overlap(x, y))
LOBPCGEigensolver.compute_λ(x::QTTBlock, hx::QTTBlock, bx::QTTBlock) =
    real.(diag(block_overlap(x, hx)) ./ diag(block_overlap(x, bx)))

function LinearAlgebra.mul!(
    destination::QTTBlock,
    source::QTTBlock,
    coefficients::AbstractMatrix,
    a::Number = true,
    b::Number = false,
)
    columns = [
        linear_expression(source.columns, a .* view(coefficients, :, j)) for
        j in axes(coefficients, 2)
    ]
    if !iszero(b)
        source.action[] == destination.action[] || error("Incompatible symbolic actions")
        columns = [
            linear_expression((columns[j], destination.columns[j]), (1, b)) for
            j in eachindex(columns)
        ]
    end
    copyto!(destination.columns, columns)
    destination.action[] = source.action[]
    destination
end
LinearAlgebra.rmul!(x::QTTBlock, coefficients::AbstractMatrix) = mul!(x, copy(x), coefficients)
LinearAlgebra.rmul!(
    x::QTTBlock,
    coefficients::Union{UpperTriangular,LowerTriangular,UnitUpperTriangular,UnitLowerTriangular},
) = mul!(x, copy(x), Matrix(coefficients))
function LinearAlgebra.mul!(destination::QTTBlock, H::BlockHamiltonian, source::QTTBlock)
    # H times a deferred search block is needed only through its projected matrix.
    copyto!(destination, source)
    destination.action[] = true
    destination
end
function LinearAlgebra.ldiv!(preconditioner::BlockPreconditioner, residuals::QTTBlock)
    orbitals = reconstruct_orbitals!(residuals; label = "lobpcg_residual")
    updated = preconditioner.apply(orbitals)
    length(updated) == length(orbitals) || throw(DimensionMismatch("Preconditioned block size"))
    residuals.columns .= [linear_expression((psi,), (1,)) for psi in updated]
    residuals
end

const QTTLazyBlock = LOBPCGEigensolver.LazyHcat{ComplexF64,<:Tuple{Vararg{QTTBlock}}}
function collect_block(x::QTTLazyBlock)
    QTTBlock(
        reduce(vcat, [block.columns for block in x.blocks]),
        first(x.blocks).workspace;
        action = first(x.blocks).action[],
    )
end
Base.:*(x::Adjoint{ComplexF64,<:QTTLazyBlock}, y::QTTBlock) =
    block_overlap(collect_block(parent(x)), y)
Base.:*(x::Adjoint{ComplexF64,<:QTTLazyBlock}, y::QTTLazyBlock) =
    block_overlap(collect_block(parent(x)), collect_block(y))
function LinearAlgebra.mul!(
    destination::QTTBlock,
    source::QTTLazyBlock,
    coefficients::AbstractMatrix,
    a::Number,
    b::Number,
)
    mul!(destination, collect_block(source), coefficients, a, b)
end
function LinearAlgebra.mul!(
    destination::QTTBlock,
    source::QTTLazyBlock,
    coefficients::AbstractMatrix,
)
    mul!(destination, source, coefficients, true, false)
    destination.action[] || reconstruct_orbitals!(destination; label = "lobpcg_update")
    destination
end
function Base.:*(source::QTTLazyBlock, coefficients::AbstractMatrix)
    destination = similar(first(source.blocks), size(source, 1), size(coefficients, 2))
    mul!(destination, source, coefficients)
end

"Ritz coefficients carry the search metric needed by the history update."
struct QTTBlockCoefficients <: AbstractMatrix{ComplexF64}
    values::Matrix{ComplexF64}
    metric::Matrix{ComplexF64}
end
Base.size(c::QTTBlockCoefficients) = size(c.values)
Base.getindex(c::QTTBlockCoefficients, i::Int, j::Int) = c.values[i, j]
function LOBPCGEigensolver.get_new_P_duersch(
    c::QTTBlockCoefficients,
    complement,
    active;
    tol,
    timer,
)
    overlap = -(complement' * c.metric[:, active])
    complement * LOBPCGEigensolver.ortho!(overlap; tol).X
end

"Use the actual search Gram after QTT reconstruction, not an assumed identity."
function LOBPCGEigensolver.rayleigh_ritz(x::QTTLazyBlock, hx::QTTLazyBlock, keep)
    # Refresh symbolic sources and release expressions from earlier iterations.
    for (source, action) in zip(x.blocks, hx.blocks)
        copyto!(action.columns, source.columns)
    end
    source, action = collect_block(x), collect_block(hx)
    projected = block_overlap(source, action)
    metric = block_metric(source)
    factor = eigen(Hermitian((projected + projected') / 2), Hermitian(metric))
    (;
        vectors = QTTBlockCoefficients(factor.vectors[:, 1:keep], metric),
        values = factor.values[1:keep],
        vectors_ortho = factor.vectors[:, (keep + 1):end],
    )
end

struct QTTBlockStyle <: Base.Broadcast.AbstractArrayStyle{2} end
Base.BroadcastStyle(::Type{<:QTTBlock}) = QTTBlockStyle()
Base.BroadcastStyle(::QTTBlockStyle, ::Base.Broadcast.DefaultArrayStyle{N}) where {N} =
    QTTBlockStyle()
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{N}, ::QTTBlockStyle) where {N} =
    QTTBlockStyle()
block_argument(x::Base.Broadcast.Extruded) = x.x
block_argument(x) = x
block_column(x::QTTBlock, j) = x.columns[j]
block_column(x::Number, j) = x
block_column(x::AbstractMatrix, j) = x[1, j]
block_column(x::Base.Broadcast.Extruded, j) = block_column(x.x, j)
function block_column(x::Base.Broadcast.Broadcasted, j)
    arguments = map(a -> block_column(a, j), x.args)
    x.f === identity && return only(arguments)
    if x.f === (*) || x.f === (/)
        left, right = arguments
        return left isa Number ? scale(right, left) : scale(left, x.f === (/) ? inv(right) : right)
    elseif x.f === (+) || x.f === (-)
        return linear_expression(arguments, (1, x.f === (-) ? -1 : 1))
    end
    error("Unsupported QTT block broadcast: $(x.f)")
end
function Base.copyto!(destination::QTTBlock, expression::Base.Broadcast.Broadcasted)
    arguments = map(block_argument, expression.args)
    if expression.f === identity && only(arguments) isa QTTBlock
        return copyto!(destination, only(arguments))
    end
    # Refresh physical residuals from stored orbitals, never from cached H combinations.
    if expression.f === (-) && first(arguments) isa QTTBlock && first(arguments).action[]
        scaled = last(arguments)
        source = block_argument(first(scaled.args))
        energies = block_argument(last(scaled.args))
        orbitals = reconstruct_orbitals!(source; label = "lobpcg_ritz")
        workspace = destination.workspace
        residuals =
            recorded_map(QTT, length(orbitals), workspace.accuracy, workspace.recorder) do i, rec
                h_residual(
                    workspace.H,
                    orbitals[i],
                    real(energies[i]),
                    workspace.accuracy,
                    rec;
                    tol = workspace.accuracy.work_tolerance,
                    cap = workspace.accuracy.work_cap,
                    potential = workspace.potential,
                    label = "lobpcg_residual_$i",
                )
            end
        destination.columns .= [linear_expression((psi,), (1,)) for psi in residuals]
        destination.action[] = false
        return destination
    end
    destination.columns .= [block_column(expression, j) for j in axes(destination, 2)]
    destination
end

function Random.randn!(rng::AbstractRNG, x::QTTBlock)
    accuracy = x.workspace.accuracy
    bonds = [min(4, 1 << min(k, accuracy.R - k)) for k = 0:(accuracy.R)]
    x.columns .= [
        linear_expression(
            (
                normalize(
                    QTT([
                        randn(rng, ComplexF64, bonds[k], 2, bonds[k + 1]) for k = 1:(accuracy.R)
                    ]),
                ),
            ),
            (1,),
        ) for _ in x.columns
    ]
    x
end

function lobpcg_audit(H, block::QTTBlock, energies)
    orbitals = reconstruct_orbitals!(block; label = "lobpcg_final")
    workspace = block.workspace
    accuracy, rec = workspace.accuracy, workspace.recorder
    residuals = recorded_map(Float64, length(orbitals), accuracy, rec) do i, localrec
        h_residual_norm(
            workspace.H,
            orbitals[i],
            energies[i],
            accuracy,
            localrec;
            tol = accuracy.algebra_tolerance,
            cap = max(accuracy.work_cap, accuracy.orbital_cap),
            potential = workspace.potential,
            label = "lobpcg_audit_$i",
        ) / ttnorm(orbitals[i])
    end
    orbitals, residuals, norm(gram(orbitals; cfg = accuracy) - I)
end

"Run the library's unchanged LOBPCG iteration on QTT block algebra."
function block_lobpcg(
    H,
    orbitals::AbstractVector{QTT},
    accuracy,
    rec;
    preconditioner = nothing,
    tolerance = accuracy.eigen_tolerance,
    work_tolerance = accuracy.algebra_tolerance,
    max_iterations = 100,
    gram_tolerance = accuracy.gram_tolerance,
    callback = identity,
)
    isempty(orbitals) && throw(ArgumentError("An initial orbital block is required"))
    validate_truncation(work_tolerance, accuracy.work_cap)
    accuracy = merge(accuracy, (; work_tolerance))
    workspace = QTTBlockWorkspace(
        accuracy,
        rec,
        H,
        synchronized(potential_samples(H, rec)),
        IdDict{Any,QTT}(),
        QTT[],
        IdDict{Any,Int}(),
        zeros(ComplexF64, 0, 0),
    )
    block = QTTBlock([linear_expression((psi,), (1,)) for psi in orbitals], workspace)
    monitor = info -> begin
        empty!(workspace.actions)
        empty!(workspace.overlap_basis)
        empty!(workspace.overlap_positions)
        workspace.overlap_metric = zeros(ComplexF64, 0, 0)
        callback(info)
    end
    block_lobpcg(
        BlockHamiltonian(H),
        block;
        preconditioner,
        tolerance,
        max_iterations,
        gram_tolerance,
        callback = monitor,
    )
end

# TT tangent-space projection and reconstruction; see Rakhuba et al., Sec. 3.
# https://arxiv.org/abs/1811.11049
# left/right hold orthogonal cores U/V; variations satisfy the left gauge condition.

struct TangentSpace
    left::Vector{Array{ComplexF64,3}}
    right::Vector{Array{ComplexF64,3}}
    shapes::Vector{Tuple{Int,Int,Int}}
    offsets::Vector{Int}
end
function tangent_space(x::QTT; cfg = (;))
    # Tangent bases need orthogonality, not another rank truncation.
    u, v = with_sampling_seed(0) do
        psi = ITensorMPS.MPS(tensortrain(x))
        ITensorMPS.orthogonalize!(psi, 1)
        right = QTT(psi).c
        ITensorMPS.orthogonalize!(psi, length(x))
        QTT(psi).c, right
    end
    d = length(x)
    shapes = [(size(u[k], 1), 2, size(v[k], 3)) for k = 1:d]
    offsets = cumsum(vcat(0, [prod(s) for s in shapes]))
    TangentSpace(u, v, shapes, offsets)
end
function tangent_cores(tangent_basis::TangentSpace, z)
    [
        reshape(
            view(z, (tangent_basis.offsets[k] + 1):tangent_basis.offsets[k + 1]),
            tangent_basis.shapes[k],
        ) for k = 1:length(tangent_basis.shapes)
    ]
end
# Remove the component parallel to the left basis to avoid redundant tangent coordinates.
function left_gauge!(g, u)
    m = reshape(g, :, size(g, 3))
    q = reshape(u, :, size(u, 3))
    m .-= q * (q' * m)
    g
end
"Project an MPO action directly into tangent coordinates, without constructing W*x."
function project(tangent_basis::TangentSpace, x::QTT, w = nothing)
    # The tangent basis and trial orbital can differ, so environments carry separate bonds.
    d = length(x)
    if w === nothing
        w = [reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1) for k = 1:d]
    end
    right = Vector{Array{ComplexF64,3}}(undef, d + 1)
    right[d + 1] = ones(ComplexF64, 1, 1, 1)
    for k = d:-1:1
        v = tangent_basis.right[k]
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        @tensor t1[l, s, I, J] := z[l, s, L] * t[I, J, L]
        @tensor t2[j, s, l, I] := a[j, s, t, J] * t1[l, t, I, J]
        @tensor e[i, j, l] := conj(v[i, s, I]) * t2[j, s, l, I]
        right[k] = e
    end
    left = ones(ComplexF64, 1, 1, 1)
    variations = Array{ComplexF64,3}[]
    for k = 1:d
        a = w[k]
        z = x.c[k]
        t = right[k + 1]
        u = tangent_basis.left[k]
        @tensor t1[i, j, s, L] := left[i, j, l] * z[l, s, L]
        @tensor t2[i, s, J, L] := t1[i, j, t, L] * a[j, s, t, J]
        @tensor g[i, s, I] := t2[i, s, J, L] * t[I, J, L]
        k < d && left_gauge!(g, u)
        push!(variations, g)
        if k < d
            @tensor e[I, J, L] := conj(u[i, s, I]) * t2[i, s, J, L]
            left = e
        end
    end
    reduce(vcat, vec.(variations))
end
function tangent_vector(tangent_basis::TangentSpace, z)
    variations = tangent_cores(tangent_basis, z)
    c = Array{ComplexF64,3}[]
    d = length(variations)
    for k = 1:d
        u = tangent_basis.left[k]
        v = tangent_basis.right[k]
        a = variations[k]
        if k == 1
            g = cat(a, u; dims = 3)
        elseif k == d
            g = cat(v, a; dims = 1)
        else
            g = zeros(ComplexF64, size(v, 1) + size(u, 1), 2, size(v, 3) + size(u, 3))
            g[1:size(v, 1), :, 1:size(v, 3)] = v
            g[(size(v, 1) + 1):end, :, 1:size(v, 3)] = a
            g[(size(v, 1) + 1):end, :, (size(v, 3) + 1):end] = u
        end
        push!(c, g)
    end
    QTT(c)
end

# 1D FD2 Hamiltonian sampling, tangent projection, and exact MPO expectation values.

"Construct and cache a diagonal potential MPO only when a core contraction needs it."
struct DeferredPotentialMPO{F}
    build::F
    cores::Base.RefValue{Union{Nothing,Vector{Array{ComplexF64,4}}}}
    lock::ReentrantLock
end
DeferredPotentialMPO(build) = DeferredPotentialMPO(
    build,
    Ref{Union{Nothing,Vector{Array{ComplexF64,4}}}}(nothing),
    ReentrantLock(),
)
potential_mpo(H, rec = nothing) = potential_mpo(H.diagonal, rec)
potential_mpo(cores::Vector{Array{ComplexF64,4}}, rec = nothing) = cores
function potential_mpo(potential::DeferredPotentialMPO, rec = nothing)
    lock(potential.lock) do
        if potential.cores[] === nothing
            potential.cores[] = with_sampling_seed(0) do
                potential.build(rec)
            end
        end
        potential.cores[]::Vector{Array{ComplexF64,4}}
    end
end

struct Hamiltonian{S,D}
    potential::Union{Nothing,QTT}
    diagonal::D
    plus::Vector{Array{ComplexF64,4}}
    minus::Vector{Array{ComplexF64,4}}
    h::Float64
    samples::S
end
Hamiltonian(v, d, plus, minus, h) = Hamiltonian(v, d, plus, minus, Float64(h), nothing)

"Evaluate a complete periodic FD2 action before fitting its QTT representation."
function hamiltonian_samples(H::Hamiltonian, orbital, potential, R; energy = 0.0)
    n = 1 << R
    sampled(R) do j
        center = orbital(j)
        kinetic = ((center - orbital(mod(j - 1, n))) + (center - orbital(mod(j + 1, n)))) / (2H.h^2)
        kinetic + (potential(j) - energy) * center
    end
end

"Share the fixed local field across all sampled actions of one Hamiltonian."
potential_samples(H::Hamiltonian, rec) =
    H.samples === nothing ? memoized_sample(H.potential, rec) : H.samples
# Used only when every initial pivot cancels. Preserve a core-algebra fallback
# instead of declaring an unsampled large-grid function globally zero.
function h_core_fallback(H, psi, energy; tol, cap, rec = nothing)
    shift_tol = min(tol, tol * H.h^2)
    terms = (
        psi,
        apply_mpo(potential_mpo(H, rec), psi; tol, cap),
        apply_mpo(H.plus, psi; tol = shift_tol, cap),
        apply_mpo(H.minus, psi; tol = shift_tol, cap),
    )
    weights = (1 / H.h^2 - energy, 1, -1 / (2H.h^2), -1 / (2H.h^2))
    reconstruct_expression(linear_expression(terms, weights); tol, cap)
end

"Construct FD2 amplitudes by differencing cores before division by the grid spacing."
function kinetic_action(
    H::Hamiltonian,
    psi,
    action_tol;
    absolute_tolerance = nothing,
    cap = nothing,
)
    if absolute_tolerance !== nothing
        # ||D|| <= 2. Allocate half the physical action error to each difference.
        gradient =
            shift_difference(H.plus, psi; cap, absolute_tolerance = absolute_tolerance * H.h^2 / 2)
        second = shift_difference(
            H.minus,
            gradient;
            cap,
            absolute_tolerance = absolute_tolerance * H.h^2,
        )
        return scale(second, 1 / (2H.h^2))
    end
    gradient = shift_difference(H.plus, psi; cap, tol = action_tol * min(1.0, H.h))
    scale(shift_difference(H.minus, gradient; cap, tol = action_tol), 1 / (2H.h^2))
end

"Sample the final unfiltered physical residual."
function h_residual(
    H,
    psi,
    energy,
    cfg,
    rec;
    tol = cfg.algebra_tolerance,
    cap = cfg.work_cap,
    label = "orbital_residual",
    potential = nothing,
)
    validate_truncation(tol, cap)
    isfinite(energy) || throw(ArgumentError("Nonfinite orbital energy"))
    orbital = sampled(psi, rec)
    field =
        potential === nothing ? potential_samples(H, rec) :
        potential isa SampledFunction ? potential :
        potential isa Evaluator ?
        TensorSamples(
            potential;
            tt_cache_capacity = get(rec.options, :slice_cache_capacity, 2048),
        ) : sampled(potential, length(psi))
    # A relative residual tolerance cannot be compared to eps/h^2 before the
    # residual scale is known. Control rounding after each core difference.
    action_tol = min(tol, cfg.algebra_tolerance)
    kinetic = kinetic_action(
        H,
        psi,
        action_tol;
        absolute_tolerance = get(cfg, :kinetic_action_tolerance, nothing),
        cap = cfg.work_cap,
    )
    fn = pointwise((t, value, v) -> t + (v - energy) * value, sampled(kinetic, rec), orbital, field)
    cross(
        rec,
        length(psi),
        fn;
        tol,
        cap,
        label,
        initial_zero_fallback = () -> h_core_fallback(H, psi, energy; tol, cap, rec),
    )
end

"Contract a residual norm without interpolating its high-frequency kinetic component."
function h_residual_norm(H::Hamiltonian, psi, energy, cfg, rec; tol, cap, label, potential)
    kinetic = kinetic_action(
        H,
        psi,
        min(tol, cfg.algebra_tolerance);
        absolute_tolerance = get(cfg, :kinetic_action_tolerance, nothing),
        cap = cfg.work_cap,
    )
    source = cross(
        rec,
        length(psi),
        pointwise((v, value) -> (v - energy) * value, potential, sampled(psi, rec));
        tol = min(tol, 0.01cfg.eigen_tolerance),
        cap,
        label = label * "_potential",
        initial_zero_fallback = () -> begin
            product = apply_mpo(potential_mpo(H, rec), psi; tol = cfg.algebra_tolerance, cap)
            reconstruct_expression(linear_expression((product, psi), (1, -energy)); tol, cap)
        end,
    )
    ttnorm(linear_expression((kinetic, source), (1, 1)))
end
h_residual_norm(H, psi, energy, cfg, rec; kwargs...) =
    ttnorm(h_residual(H, psi, energy, cfg, rec; kwargs...))

function h_action(H, psi, cfg, rec; label = "", work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid H-action tolerance"))
    tol = work_tol > 0 ? work_tol : cfg.algebra_tolerance
    h_residual(
        H,
        psi,
        0.0,
        cfg,
        rec;
        tol,
        potential,
        label = (work_tol > 0 ? "Hpsi_work_" : "Hpsi_") * label,
    )
end
function h_project(tangent_basis, psi, H; projected_psi = nothing)
    center = projected_psi === nothing ? project(tangent_basis, psi) : projected_psi
    project(tangent_basis, psi, potential_mpo(H)) +
    kinetic_project(tangent_basis, psi, H.plus, H.minus, H.h, center)
end

"Use the gradient form when subtracting unit-scale shift projections loses digits."
function kinetic_project(tangent_basis, psi, plus, minus, h, center)
    left, right = project(tangent_basis, psi, minus), project(tangent_basis, psi, plus)
    numerator = ((center - left) + (center - right)) / 2
    if norm(numerator) <= sqrt(eps(Float64)) * (norm(center) + (norm(left) + norm(right)) / 2)
        gradient = shift_difference(plus, psi)
        return (project(tangent_basis, gradient, minus) - project(tangent_basis, gradient)) / (2h^2)
    end
    numerator / h^2
end

"Project H on gauge-constrained tangent coordinates, such as L4's projected/SVD columns."
function h_project_tangent(tangent_basis, z, H)
    tangent = tangent_vector(tangent_basis, z)
    h_project(tangent_basis, tangent, H; projected_psi = z)
end
function h_expectation(H, psi)
    # All coefficient-Hamiltonian blocks must use the same MPO contraction.
    # A loose TCI Rayleigh diagonal becomes an artificial negative direction
    # when the joint old-orbital / tangent Gram matrix is nearly singular.
    real(only(h_gram([psi], H)))
end

"Kinetic expectations from shared MPO contractions, with a stable gradient fallback."
function kinetic_expectations(orbitals, plus, minus, h; cfg = (;))
    b = length(orbitals)
    out = zeros(Float64, b)
    with_sampling_seed(0) do
        bras, kets, (shift_plus, shift_minus) = mpo_basis(orbitals, (plus, minus))
        foreach_independent(b, cfg) do i
            overlap = inner(orbitals[i], orbitals[i]; cfg)
            terms = (
                overlap,
                -ITensorMPS.inner(bras[i], shift_plus, kets[i]; make_inds_match = false) / 2,
                -ITensorMPS.inner(bras[i], shift_minus, kets[i]; make_inds_match = false) / 2,
            )
            numerator = compensated_sum(terms)
            # Nearly equal FD2 terms lose digits; use the exact low-rank gradient.
            if abs(numerator) <= sqrt(eps(Float64)) * sum(abs, terms) || real(numerator) < 0
                gradient = shift_difference(plus, orbitals[i])
                numerator = inner(gradient, gradient; cfg) / 2
            end
            # numerator already contains the 1/2 from (plus+minus)/2.
            out[i] = real(numerator) / h^2
        end
    end
    out
end

function shift_difference(shift, psi; tol = 0.0, absolute_tolerance = nothing, cap = nothing)
    # Apply the shift exactly; round only after cancellation in the common QR basis.
    # Energy contractions use the default zero tolerance.
    exact_cap = maximum(size(w, 4) * size(g, 3) for (w, g) in zip(shift, psi.c))
    shifted = apply_mpo(shift, psi; tol = 0.0, cap = exact_cap)
    reconstruct_expression(
        linear_expression((shifted, psi), (1, -1));
        tol,
        cap = cap === nothing ? rank(shifted) + rank(psi) : cap,
        absolute_tolerance,
    )
end

"Project H from cores; use stable gradient contractions when FD2 terms cancel."
function h_gram(orbitals, H; cfg = (;))
    with_sampling_seed(0) do
        b = length(orbitals)
        bras, kets, (potential, plus, minus) =
            mpo_basis(orbitals, (potential_mpo(H), H.plus, H.minus))
        overlaps = gram(orbitals; cfg)
        A = zeros(ComplexF64, b, b)
        fallback = fill(false, b, b)
        foreach_independent(b, cfg) do j
            for i = 1:j
                terms = (
                    overlaps[i, j],
                    -ITensorMPS.inner(bras[i], plus, kets[j]; make_inds_match = false) / 2,
                    -ITensorMPS.inner(bras[i], minus, kets[j]; make_inds_match = false) / 2,
                )
                numerator = compensated_sum(terms)
                fallback[i, j] =
                    abs(numerator) <= sqrt(eps(Float64)) * sum(abs, terms) ||
                    (i == j && real(numerator) < 0)
                A[i, j] =
                    ITensorMPS.inner(bras[i], potential, kets[j]; make_inds_match = false) +
                    (fallback[i, j] ? 0 : numerator / H.h^2)
            end
        end
        needed = [i for i = 1:b if any(fallback[min(i, j), max(i, j)] for j = 1:b)]
        gradients = Vector{QTT}(undef, b)
        foreach_independent(length(needed), cfg) do k
            i = needed[k]
            gradients[i] = shift_difference(H.plus, orbitals[i])
        end
        foreach_independent(b, cfg) do j
            for i = 1:j
                fallback[i, j] && (A[i, j] += inner(gradients[i], gradients[j]; cfg) / (2H.h^2))
                A[j, i] = conj(A[i, j])
                i == j && (A[i, i] = real(A[i, i]))
            end
        end
        require_finite(A, "nonfinite_direct_hamiltonian_gram")
        A
    end
end

# Orbital updates, orthogonalization, and density mixing.

# Keep consecutive orbital transforms in coefficient space until cores are needed.

mutable struct OrbitalTransform
    basis::Vector{QTT}
    coefficients::Matrix{ComplexF64}
    metric::Matrix{ComplexF64}
    transforms::Int
    rotated::Bool
end
function OrbitalTransform(orbitals; cfg = (;), metric = gram(orbitals; cfg))
    OrbitalTransform(
        collect(orbitals),
        Matrix{ComplexF64}(I, length(orbitals), length(orbitals)),
        metric,
        0,
        false,
    )
end
transform_gram(x::OrbitalTransform) = x.coefficients' * x.metric * x.coefficients

function orthogonalize_coefficients!(x::OrbitalTransform, cfg; force = false)
    S = Hermitian(transform_gram(x))
    require_finite(Matrix(S), "nonfinite_orbital_transform_gram")
    defect = norm(S - I)
    defect < cfg.orthogonalize_threshold && !force && return false
    fac = eigen(S)
    minimum(fac.values) > cfg.independence_threshold ||
        error("Occupied orbitals lost linear independence")
    x.coefficients =
        x.coefficients * (fac.vectors * Diagonal(1 ./ sqrt.(fac.values)) * fac.vectors')
    x.transforms += 1
    true
end

function ritz_projection(orbitals, H, cfg, rec; work_tol = 0.0, potential = nothing)
    stage = work_tol > 0 ? "ritz_sampled_projection" : "ritz_direct_projection"
    timed(rec, stage) do
        work_tol > 0 || return h_gram(orbitals, H; cfg)
        # Apply H once per input column, at the existing work tolerance/cap.
        # The following coefficient transforms reuse this fixed-basis matrix.
        field = synchronized(potential === nothing ? potential_samples(H, rec) : potential)
        hx = recorded_map(QTT, length(orbitals), cfg, rec) do i, localrec
            h_action(H, orbitals[i], cfg, localrec; label = "ritz_$i", work_tol, potential = field)
        end
        A = gram(orbitals, hx; cfg)
        require_finite(A, "nonfinite_sampled_ritz_projection")
        (A + A') / 2
    end
end
function ritz_coefficients!(
    x::OrbitalTransform,
    H,
    cfg,
    rec;
    force = false,
    work_tol = 0.0,
    potential = nothing,
)
    projected = ritz_projection(x.basis, H, cfg, rec; work_tol, potential)
    A = x.coefficients' * projected * x.coefficients
    A = (A + A') / 2
    require_finite(A, "nonfinite_orbital_transform_hamiltonian")
    offdiag = norm(A - Diagonal(diag(A)))
    !force && offdiag < cfg.eigen_tolerance / 10 && return offdiag
    x.coefficients = x.coefficients * eigen(Hermitian(A)).vectors
    squared_norms = real.(diag(transform_gram(x)))
    all(n -> isfinite(n) && n > 0, squared_norms) || error("Invalid rotated orbital norm")
    x.coefficients ./= transpose(sqrt.(squared_norms))
    x.transforms += 1
    x.rotated = true
    offdiag
end

function reconstruct_orbitals(x::OrbitalTransform, cfg, rec; label, combination_samples = nothing)
    group_size = get(rec.options, :orbital_pivot_group_size, 4)
    evaluators =
        combination_samples === nothing ?
        orbital_evaluators(x.basis, size(x.coefficients, 2), cfg, rec; group_size) : nothing
    updated_orbitals =
        recorded_map(QTT, size(x.coefficients, 2), cfg, rec; group_size) do i, localrec
            y = linear_combination(
                localrec,
                x.basis,
                view(x.coefficients, :, i);
                evaluator = evaluators === nothing ? nothing : orbital_evaluator(evaluators),
                samples = combination_samples === nothing ? nothing :
                          combination_samples(view(x.coefficients, :, i)),
                tol = cfg.orthogonalization_tolerance,
                cap = cfg.orbital_cap,
                label = "$(label)_$i",
                shared_pivots = "orbital_basis_group_$(cld(i, group_size))",
                require_convergence = get(cfg, :adaptive_accuracy, false),
            )
            x.rotated ? normalize(y) : y
        end
    updated_orbitals, evaluators === nothing ? (;) : block_cache_stats(evaluators)
end

function finish_orbital_transform(
    x::OrbitalTransform,
    original,
    cfg,
    rec;
    combination_samples = nothing,
)
    x.transforms == 0 && return original, norm(x.metric - I)
    label = x.rotated ? "ritz_rotate" : "orthogonalize"
    updated_orbitals, cache_stats = reconstruct_orbitals(x, cfg, rec; label, combination_samples)
    metric = gram(updated_orbitals; cfg)
    defect = norm(metric - I)
    repaired = false
    # Compression is nonlinear: a coefficient-space identity does not certify
    # the fitted cores. Retain the original one-pass post-Ritz repair if needed.
    if x.rotated && defect >= cfg.orthogonalize_threshold
        updated_orbitals, defect = orthogonalize(updated_orbitals, cfg, rec; metric)
        repaired = true
    end
    event!(
        rec.stage_io,
        (;
            scf = rec.scf,
            inner = rec.inner,
            stage = "orbital_transform",
            parent_stage = rec.stage,
            seconds = 0.0,
            gc_seconds = 0.0,
            allocated_bytes = 0,
            tci_calls = 0,
            tci_samples = 0,
            tci_seconds = 0.0,
            event = "orbital_transform",
            basis_columns = length(x.basis),
            coefficient_transforms = x.transforms,
            reconstruction_passes = 1 + Int(repaired),
            post_fit_orthogonalization = repaired,
            gram_defect = defect,
            cache_stats...,
        ),
    )
    updated_orbitals, defect
end

function orthogonalize(
    orbitals,
    cfg,
    rec;
    force = false,
    metric = gram(orbitals; cfg),
    combination_samples = nothing,
)
    transform = OrbitalTransform(orbitals; cfg, metric)
    orthogonalize_coefficients!(transform, cfg; force)
    finish_orbital_transform(transform, orbitals, cfg, rec; combination_samples)
end

function ritz_rotate(orbitals, H, cfg, rec; work_tol = 0.0, force = false)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(orbitals; cfg)
    offdiag = ritz_coefficients!(transform, H, cfg, rec; force, work_tol)
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    updated_orbitals, _ = finish_orbital_transform(transform, orbitals, cfg, rec)
    updated_orbitals, offdiag
end

"Orthogonalize, optionally rotate, and rebuild only the final occupied columns."
function orthogonalize_ritz(orbitals, H, cfg, rec; ritz = true, work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 || throw(ArgumentError("Invalid Ritz work tolerance"))
    transform = OrbitalTransform(orbitals; cfg)
    orthogonalize_coefficients!(transform, cfg)
    offdiag = ritz ? ritz_coefficients!(transform, H, cfg, rec; work_tol, potential) : 0.0
    transform.rotated && orthogonalize_coefficients!(transform, cfg)
    updated_orbitals, defect = finish_orbital_transform(transform, orbitals, cfg, rec)
    updated_orbitals, defect, offdiag
end

# Optimize small coefficient matrices while enforcing orthogonality between orbitals.

"Alternating minimization with diagonal coefficients on old orbitals (Rakhuba et al., Eqs. (23)-(29))."
function coefficients(
    S_xx,
    S_vx,
    H_vx,
    H_vv,
    energies;
    nsweeps = 2,
    parallel_tasks = 1,
    metric_relative_cutoff = 1e-11,
    metric_absolute_cutoff = 1e-13,
    constraint_relative_cutoff = 1e-10,
    constraint_absolute_cutoff = 1e-12,
)
    all(isfinite, S_xx) &&
        all(isfinite, S_vx) &&
        all(isfinite, H_vx) &&
        all(isfinite, H_vv) &&
        all(isfinite, energies) || error("Nonfinite input to coefficient minimizer")
    # X contains orbitals and V orthonormal tangent directions: S_vx = V'X, H_vx = V'HX.
    # S_xx = X'X and H_vv = V'HV; c and C follow the paper's coefficient notation.
    m, b = size(S_vx)
    G = [S_xx S_vx'; S_vx Matrix{ComplexF64}(I, m, m)]
    coef = vcat(Matrix{ComplexF64}(I, b, b), zeros(ComplexF64, m, b))
    orthogonalizers = Vector{Matrix{ComplexF64}}(undef, b)
    hamiltonians = Vector{Matrix{ComplexF64}}(undef, b)
    foreach_independent(b, (; parallel_tasks)) do i
        local ids, a
        ids = vcat(i, (b + 1):(b + m))
        # Preserve the old orbital exactly when removing near-dependent tangent
        # directions; truncating the joint Gram eigenvectors can discard it.
        xnorm = sqrt(real(S_xx[i, i]))
        overlap = S_vx[:, i] / xnorm
        e = eigen(Hermitian(Matrix{ComplexF64}(I, m, m) - overlap * overlap'))
        good = findall(
            >(max(maximum(e.values; init = 0.0) * metric_relative_cutoff, metric_absolute_cutoff)),
            e.values,
        )
        tangent_coefficients = e.vectors[:, good] * Diagonal(1 ./ sqrt.(e.values[good]))
        orthogonalizer = zeros(ComplexF64, m + 1, length(good) + 1)
        orthogonalizer[1, 1] = inv(xnorm)
        orthogonalizer[1, 2:end] = vec(-(overlap' * tangent_coefficients) / xnorm)
        orthogonalizer[2:end, 2:end] = tangent_coefficients
        orthogonalizers[i] = orthogonalizer
        a = zeros(ComplexF64, m + 1, m + 1)
        a[1, 1] = energies[i] * real(S_xx[i, i])
        a[2:end, 1] = H_vx[:, i]
        a[1, 2:end] = conj.(H_vx[:, i])
        a[2:end, 2:end] = H_vv
        hamiltonians[i] = a
    end
    # Each orbital's constraints use the coefficients already updated in this sweep.
    for sweep = 1:nsweeps, i = 1:b
        ids = vcat(i, (b + 1):(b + m))
        other = [j for j = 1:b if j != i]
        orthogonalizer = orthogonalizers[i]
        constraints = orthogonalizer' * (G[ids, :] * coef[:, other])
        s = checked_svd(constraints; full = true)
        constraint_rank = count(
            >(
                max(
                    isempty(s.S) ? 0.0 : maximum(s.S) * constraint_relative_cutoff,
                    constraint_absolute_cutoff,
                ),
            ),
            s.S,
        )
        null_basis = s.U[:, (constraint_rank + 1):end]
        size(null_basis, 2) > 0 || error("No feasible coefficient direction")
        feasible_basis = orthogonalizer * null_basis
        a = Hermitian(feasible_basis' * hamiltonians[i] * feasible_basis)
        all(isfinite, a) || error(
            "Nonfinite reduced coefficient matrix at sweep=$sweep orbital=$i; orthogonalizer_max=$(maximum(abs,orthogonalizer)) nullspace_size=$(size(null_basis))",
        )
        ev = eigen(a)
        v = feasible_basis * ev.vectors[:, 1]
        overlap = dot(G[ids, i], v)
        abs(overlap) > 1e-12 && (v .*= conj(overlap) / abs(overlap))
        coef[:, i] .= 0
        coef[ids, i] = v
    end
    orbital_energies = [
        real(
            dot(
                coef[vcat(i, (b + 1):(b + m)), i],
                hamiltonians[i] * coef[vcat(i, (b + 1):(b + m)), i],
            ),
        ) for i = 1:b
    ]
    (
        c = diag(coef[1:b, :]),
        C = coef[(b + 1):end, :],
        gram_defect = norm(coef' * G * coef - I),
        trace_after = sum(orbital_energies),
        orbital_energies = orbital_energies,
    )
end

# Low-rank orbital update; stages L1 to L7 match the performance logs.

"Green fixed-point update: (T + mu*I)^(-1) * (mu + epsilon - V) * psi."
function green_rhs(H, psi, energy, cfg, rec; potential, label)
    cross(
        rec,
        cfg.R,
        pointwise(
            (v, value) -> (cfg.green_shift + energy - v) * value,
            potential,
            sampled(psi, rec),
        );
        tol = cfg.green_rhs_tolerance,
        cap = get(cfg, :green_rhs_cap, cfg.work_cap),
        label,
    )
end

function block_green(orbitals, energies, H, green, cfg, rec; potential = nothing)
    potential_samples_shared =
        synchronized(potential === nothing ? potential_samples(H, rec) : potential)
    recorded_map(QTT, length(orbitals), cfg, rec) do i, localrec
        rhs = green_rhs(
            H,
            orbitals[i],
            energies[i],
            cfg,
            localrec;
            potential = potential_samples_shared,
            label = "block_green_$i",
        )
        # Truncated MPO action establishes its own QR gauge when needed.
        normalize(
            apply_mpo(
                green,
                rhs;
                tol = min(cfg.algebra_tolerance, cfg.orbital_tolerance),
                cap = cfg.orbital_cap,
                rec = localrec,
            ),
        )
    end
end
function lrrap_step(orbitals, previous, H, green, cfg, rec; work_tol = 0.0, potential = nothing)
    isfinite(work_tol) && work_tol >= 0 ||
        throw(ArgumentError("work_tol must be finite and nonnegative"))
    algebra_tol = work_tol > 0 ? work_tol : cfg.algebra_tolerance
    state_tol = min(algebra_tol, cfg.orbital_tolerance)
    b = length(orbitals)
    field = potential === nothing ? potential_samples(H, rec) : potential
    # L1 needs Rayleigh energies and residual norms, not residual tensor fits.
    energies, residual_norms = timed(rec, "L1") do
        potential_samples_shared = synchronized(field)
        results = recorded_map(Tuple{Float64,Float64}, b, cfg, rec) do i, localrec
            e = h_expectation(H, orbitals[i]) / real(inner(orbitals[i], orbitals[i]))
            r = h_residual_norm(
                H,
                orbitals[i],
                e,
                cfg,
                localrec;
                tol = algebra_tol,
                cap = cfg.work_cap,
                label = "orbital_residual_$i",
                potential = potential_samples_shared,
            )
            e, r
        end
        first.(results), last.(results)
    end
    # Cycle the tangent-space base point through the occupied orbitals.
    tangent_orbital = mod((rec.scf - 1) * cfg.inner_steps + rec.inner - 1, b) + 1
    tangent_basis, tangent_base_rank = timed(rec, "L2") do
        base = orbitals[tangent_orbital]
        if work_tol > 0
            # The trial tangent space uses the work budget; stored orbitals retain
            # their finer representation and enter S_xx/H_vx without this rounding.
            base = rounded(base; tol = min(work_tol, cfg.green_rhs_tolerance), cap = cfg.work_cap)
        end
        tangent_space(base; cfg), rank(base)
    end
    # L3: project orbitals, Green-preconditioned residuals, and prior directions into one tangent space.
    DX, DZ, DP = timed(rec, "L3") do
        rows = tangent_basis.offsets[end]
        dx = independent_columns(i -> project(tangent_basis, orbitals[i]), rows, b, cfg)
        # G*(H-epsilon)*psi = psi - G*(mu+epsilon-V)*psi.
        # Fit the smooth source before applying G, avoiding a noisy FD2 intermediate.
        shared_field = synchronized(field)
        preconditioned = recorded_map(Vector{ComplexF64}, b, cfg, rec) do i, localrec
            rhs = green_rhs(
                H,
                orbitals[i],
                energies[i],
                cfg,
                localrec;
                potential = shared_field,
                label = "green_direction_rhs_$i",
            )
            view(dx, :, i) - project(tangent_basis, rhs, green)
        end
        dz = reduce(hcat, preconditioned)
        dp = independent_columns(
            i -> project(tangent_basis, previous[i]),
            rows,
            previous === nothing ? 0 : length(previous),
            cfg,
        )
        dx, dz, dp
    end
    # Only tangent coordinates are needed beyond this point.
    D, W, H_vv, S_vx, H_vx, S_xx = timed(rec, "L4") do
        raw = hcat(DX, DZ, DP)
        # Orthonormalize the tangent coordinates with SVD. X retains diagonal coefficients.
        column_norms = [max(norm(view(raw, :, j)), 1e-14) for j = 1:size(raw, 2)]
        raw ./= transpose(column_norms)
        # Overwrite the normalized coordinate buffer and retain a view of U.
        # This avoids three extra copies of the largest tangent-coordinate array.
        fac = svd!(raw; alg = LinearAlgebra.QRIteration())
        good = findall(
            >(max(maximum(fac.S) * cfg.tangent_relative_cutoff, cfg.tangent_absolute_cutoff)),
            fac.S,
        )
        W = Diagonal(1 ./ column_norms) * fac.V[:, good] * Diagonal(1 ./ fac.S[good])
        D = view(fac.U, :, 1:length(good))
        projected_actions = independent_columns(size(D, 1), size(D, 2), cfg) do j
            h_project_tangent(tangent_basis, view(D, :, j), H)
        end
        H_vv = D' * projected_actions
        H_vv = (H_vv + H_vv') / 2
        # Exact contractions with H, independently of the TCI approximation used in L1.
        phx = independent_columns(size(D, 1), b, cfg) do i
            h_project(tangent_basis, orbitals[i], H; projected_psi = view(DX, :, i))
        end
        D, W, H_vv, D' * DX, D' * phx, gram(orbitals; cfg)
    end
    # Canonical orthogonalization amplifies Hamiltonian roundoff by 1/lambda(S).
    # Resolve overlap directions only while that estimate stays below the eigen tolerance.
    coefficient_metric_cutoff =
        get(cfg, :adaptive_accuracy, false) ?
        max(
            cfg.metric_relative_cutoff,
            eps(Float64) * cfg.kinetic_spectral_bound / cfg.eigen_tolerance,
        ) : cfg.metric_relative_cutoff
    coefficient_metric_cutoff < 1 || error("Grid exceeds the Float64 coefficient resolution budget")
    # L5: optimize the small coefficient problem under the orbital orthogonality constraints.
    coefficient_solution = timed(rec, "L5") do
        debug_snapshot(
            rec,
            "coefficient_debug.jls",
            (S_xx = S_xx, S_vx = S_vx, H_vx = H_vx, H_vv = H_vv, energies = energies),
        )
        coefficients(
            S_xx,
            S_vx,
            H_vx,
            H_vv,
            energies;
            parallel_tasks = get(cfg, :parallel_tasks, 1),
            nsweeps = cfg.coefficient_sweeps,
            metric_relative_cutoff = coefficient_metric_cutoff,
            metric_absolute_cutoff = cfg.metric_absolute_cutoff,
            constraint_relative_cutoff = cfg.constraint_relative_cutoff,
            constraint_absolute_cutoff = cfg.constraint_absolute_cutoff,
        )
    end
    green_update =
        (cfg.green_update_interval > 0 && rec.scf % cfg.green_update_interval == 0) ||
        maximum(residual_norms) < cfg.green_update_residual_threshold
    # L6: reconstruct orbitals. A Green update resets the previous search directions.
    updated_orbitals, search_directions = timed(rec, "L6") do
        correction_coordinates = D * coefficient_solution.C
        history_coordinates = if green_update
            nothing
        else
            basis_coefficients = W * coefficient_solution.C
            directions = DZ * basis_coefficients[(b + 1):(2b), :]
            previous !== nothing && (directions .+= DP * basis_coefficients[(2b + 1):end, :])
            directions
        end
        results = recorded_map(Tuple{QTT,QTT}, b, cfg, rec) do i, localrec
            t = tangent_vector(tangent_basis, view(correction_coordinates, :, i))
            y = normalize(
                reconstruct_expression(
                    linear_expression((orbitals[i], t), (coefficient_solution.c[i], 1));
                    tol = state_tol,
                    cap = cfg.orbital_cap,
                    absolute_tolerance = get(cfg, :adaptive_accuracy, false) ? state_tol : nothing,
                ),
            )
            p = if green_update
                constant_tt(cfg.R, 0.0)
            else
                with_svd_context(;
                    orbital = i,
                    role = "history_direction",
                    coordinates = view(history_coordinates, :, i),
                ) do
                    rounded(
                        tangent_vector(tangent_basis, view(history_coordinates, :, i));
                        tol = algebra_tol,
                        cap = cfg.orbital_cap,
                    )
                end
            end
            y, p
        end
        first.(results), last.(results)
    end
    if green_update
        # Accepted orbitals keep the original tight Green output accuracy;
        # the looser work budget applies to search directions, not this state.
        updated_orbitals = timed(rec, "L6G") do
            # The coefficient solve already supplies Rayleigh values for these updated orbitals.
            block_green(
                updated_orbitals,
                coefficient_solution.orbital_energies,
                H,
                green,
                cfg,
                rec;
                potential = field,
            )
        end
    end
    updated_orbitals, gramdef, ritz_offdiag = timed(rec, "L7") do
        orthogonalize_ritz(
            updated_orbitals,
            H,
            cfg,
            rec;
            ritz = green_update,
            work_tol,
            potential = field,
        )
    end
    metrics = (
        tangent_orbital = tangent_orbital,
        tangent_base_rank = tangent_base_rank,
        tangent_dimension = size(D, 2),
        residual_max = maximum(residual_norms),
        residual_rms = sqrt(sum(abs2, residual_norms) / b),
        trace_before = sum(energies),
        trace_after = coefficient_solution.trace_after,
        coefficient_metric_cutoff,
        coefficient_gram_defect = coefficient_solution.gram_defect,
        gram_defect = gramdef,
        green_update = green_update,
        green_energy_update_max_Ha = maximum(
            abs.(coefficient_solution.orbital_energies .- energies),
        ),
        ritz_offdiagonal_before = ritz_offdiag,
        max_rank = maximum(rank.(updated_orbitals)),
        inner_output_max_rank = maximum(rank.(updated_orbitals)),
        orbital_bytes = sum(storage, updated_orbitals),
        history_bytes = sum(storage, search_directions),
        work_tolerance = work_tol,
        work_rank_cap = cfg.work_cap,
    )
    updated_orbitals, search_directions, metrics
end

# Density mixing and output compression use separate working and storage tolerances.

function mix_density(previous_density, orbital_density, inputs, residuals, cfg, rec)
    previous_density === nothing && return orbital_density, "initial"
    residual = linear_expression((orbital_density, previous_density), (1, -1))
    push!(inputs, previous_density)
    push!(residuals, residual)
    length(inputs) > cfg.anderson_history && (popfirst!(inputs); popfirst!(residuals))
    m = length(inputs)
    coeff = zeros(m)
    coeff[end] = 1
    method = "linear"
    if cfg.anderson && m >= 3
        B = timed(rec, "mixing_residual_gram") do
            real.(expression_gram(residuals; rec))
        end
        B .+= Diagonal(fill(maximum(diag(B)) * cfg.anderson_regularization + 1e-20, m))
        trial = B \ ones(m)
        trial ./= sum(trial)
        if all(isfinite, trial) && maximum(abs, trial) < cfg.anderson_coefficient_limit
            coeff = trial
            method = "anderson"
        end
    end
    active = findall(c -> abs(c) >= 1e-15, coeff)
    terms = vcat(inputs[active], residuals[active])
    weights = vcat(coeff[active], cfg.mixing .* coeff[active])
    # Flatten n_i + alpha*(rho_i-n_i); history holds only original QTT
    # terms, so expression depth stays one and at most 2*history terms live.
    linear_expression(terms, weights), method
end

reconstruct_density(rho::QTT, cfg, rec) = rho
function reconstruct_density(rho::QTTExpression, cfg, rec)
    timed(rec, "density_mixing") do
        mixed = reconstruct_expression(
            rho;
            tol = get(cfg, :mixing_tolerance, cfg.algebra_tolerance),
            cap = get(cfg, :density_cap, cfg.effective_potential_cap),
        )
        dx = get(cfg, :volume_element, cfg.cell / 2^cfg.R)
        charge = dx * real(sum(tensortrain(mixed)))
        isfinite(charge) && charge > 0 || error("Invalid charge in mixed density: $charge")
        scale(mixed, cfg.electrons / charge)
    end
end

"Compress stored orbitals at their requested accuracy; internal operator algebra keeps its tighter tolerance."
function compress_orbitals(orbitals, cfg; tol = cfg.orbital_tolerance, input_gram_defect = nothing)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("orbital_tol must be finite and nonnegative"))
    before = sum(storage, orbitals)
    input_defect = input_gram_defect === nothing ? norm(gram(orbitals; cfg) - I) : input_gram_defect
    tol == 0 && return orbitals,
    (
        orbital_storage_tolerance = 0.0,
        orbital_rank_before_compression = maximum(rank.(orbitals)),
        stored_max_rank = maximum(rank.(orbitals)),
        orbital_bytes_before_compression = before,
        orbital_compression_attempts = 0,
        stored_gram_defect = input_defect,
    )
    for attempt = 1:(cfg.storage_compression_attempts)
        actual = tol / 10^(attempt - 1)
        updated_orbitals = Vector{QTT}(undef, length(orbitals))
        foreach_independent(length(orbitals), cfg) do i
            updated_orbitals[i] =
                normalize(rounded(orbitals[i]; tol = actual, cap = cfg.orbital_cap))
        end
        # Compression must not sacrifice occupied-space orthogonality.
        defect = norm(gram(updated_orbitals; cfg) - I)
        representation_error =
            get(cfg, :adaptive_accuracy, false) ?
            maximum(
                independent_map(
                    i -> difference_norm(updated_orbitals[i], orbitals[i]),
                    Float64,
                    length(orbitals),
                    cfg,
                ),
            ) : nothing
        accurate = representation_error === nothing || representation_error <= 2tol + 16eps(Float64)
        if defect <= max(cfg.storage_gram_tolerance, 2input_defect) && accurate
            return updated_orbitals,
            (
                orbital_storage_tolerance = actual,
                orbital_rank_before_compression = maximum(rank.(orbitals)),
                stored_max_rank = maximum(rank.(updated_orbitals)),
                orbital_bytes_before_compression = before,
                orbital_compression_attempts = attempt,
                stored_gram_defect = defect,
                storage_orbital_error_max = representation_error,
            )
        end
    end
    error("Orbital compression failed the Gram or representation-error budget")
end

# VectorInterface keeps solver tolerances and rank caps separate from the stored QTT type.
mutable struct QTTVector
    psi::Union{QTT,QTTExpression}
    tol::Float64
    cap::Int
    rec::Recorder
end
VI.scalartype(::Type{QTTVector}) = ComplexF64
function reconstruct_orbital!(x::QTTVector)
    if x.psi isa QTTExpression
        expression = x.psi
        x.psi =
            isempty(expression.terms) ? constant_tt(expression.R, 0) :
            linear_combination(
                x.rec,
                expression.terms,
                expression.coefficients;
                tol = x.tol,
                cap = x.cap,
                label = "krylov_combination",
            )
    end
    x.psi::QTT
end
VI.inner(x::QTTVector, y::QTTVector) = x === y ? ComplexF64(norm(x)^2) : inner(x.psi, y.psi)
LinearAlgebra.norm(x::QTTVector) = ttnorm(reconstruct_orbital!(x))
VI.zerovector(x::QTTVector, ::Type{<:Number}) =
    QTTVector(linear_expression((x.psi,), (0,)), x.tol, x.cap, x.rec)
VI.zerovector!!(x::QTTVector) = VI.zerovector(x)
VI.scale(x::QTTVector, a::Number) =
    QTTVector(linear_expression((x.psi,), (a,)), x.tol, x.cap, x.rec)
VI.scale!!(x::QTTVector, a::Number) = VI.scale(x, a)
VI.scale!!(::QTTVector, x::QTTVector, a::Number) = VI.scale(x, a)
function VI.add(y::QTTVector, x::QTTVector, a::Number, b::Number)
    iszero(a) && return VI.scale(y, b)
    iszero(b) && return VI.scale(x, a)
    # Accumulate projection coefficients; fit once when a norm or operator needs cores.
    psi = linear_expression((y.psi, x.psi), (b, a))
    QTTVector(psi, y.tol, y.cap, y.rec)
end
VI.add!!(y::QTTVector, x::QTTVector, a::Number, b::Number) = VI.add(y, x, a, b)

function lanczos_orbitals(factorization, norbitals, cfg, rec; tol)
    dimension = length(factorization)
    projected =
        factorization isa KrylovKit.LanczosFactorization ?
        KrylovKit.rayleighquotient(factorization) :
        Hermitian(view(factorization.H, 1:dimension, 1:dimension))
    spectrum = eigen(projected)
    basis = reconstruct_orbital!.(KrylovKit.basis(factorization))
    orbitals = combine_orbitals(
        basis,
        spectrum.vectors[:, 1:norbitals],
        cfg,
        rec;
        tol,
        cap = cfg.orbital_cap,
        label = "lanczos_ritz",
    )
    # Repair the returned orbitals without changing the retained recurrence basis.
    orthogonalize(orbitals, cfg, rec)
end

"Advance Lanczos; reconstruct Ritz orbitals only when a caller or restart needs them."
function lanczos_step(
    orbitals,
    subspace,
    H,
    cfg,
    rec;
    solver = :lanczos,
    work_tol,
    ritz_tol,
    potential,
    extract_ritz = true,
)
    solver in (:lanczos, :block_lanczos) || throw(ArgumentError("Unknown Lanczos method"))
    norbitals = length(orbitals)
    block_recurrence = solver == :block_lanczos || norbitals > 1
    limit = min(2^cfg.R, max(16, 4norbitals))
    width =
        subspace === nothing ? norbitals :
        block_recurrence ? length(KrylovKit.residual(subspace.factorization)) : 1
    residual_floor = max(work_tol, 100eps(Float64))
    breakdown =
        subspace !== nothing &&
        KrylovKit.normres(subspace.factorization) <= sqrt(width) * residual_floor
    restarted = subspace !== nothing && !breakdown && length(subspace.factorization) + width > limit
    initialized = subspace === nothing || restarted
    if restarted && subspace.ritz_dimension != length(subspace.factorization)
        orbitals, _ = lanczos_orbitals(subspace.factorization, norbitals, cfg, rec; tol = ritz_tol)
    end
    if initialized
        action =
            x -> QTTVector(
                h_action(
                    H,
                    reconstruct_orbital!(x),
                    cfg,
                    rec;
                    label = "lanczos_action",
                    work_tol,
                    potential,
                ),
                work_tol,
                cfg.work_cap,
                rec,
            )
        initial = [QTTVector(psi, work_tol, cfg.work_cap, rec) for psi in orbitals]
        # A block preserves independent directions within degenerate eigenspaces.
        iterator = if !block_recurrence
            KrylovKit.LanczosIterator(action, only(initial), KrylovKit.ModifiedGramSchmidt2())
        else
            KrylovKit.BlockLanczosIterator(
                action,
                KrylovKit.Block(initial),
                limit,
                KrylovKit.ModifiedGramSchmidt2(),
                residual_floor,
            )
        end
        factorization = KrylovKit.initialize(iterator)
        length(factorization) >= norbitals || error("The initial Lanczos block lost occupied rank")
        subspace = (; iterator, factorization, gram_defect = nothing, ritz_dimension = 0)
    end
    factorization = subspace.factorization
    previous_dimension = length(factorization)
    width = block_recurrence ? length(KrylovKit.residual(factorization)) : 1
    # Do not normalize a numerically zero residual at an invariant subspace.
    breakdown_tolerance = sqrt(width) * residual_floor
    breakdown = KrylovKit.normres(factorization) <= breakdown_tolerance
    if !breakdown && previous_dimension + width <= limit
        KrylovKit.expand!(subspace.iterator, factorization)
    end
    dimension = length(factorization)
    basis = reconstruct_orbital!.(KrylovKit.basis(factorization))
    if extract_ritz && subspace.ritz_dimension != dimension
        orbitals, gram_defect = lanczos_orbitals(factorization, norbitals, cfg, rec; tol = ritz_tol)
        subspace = (; subspace.iterator, factorization, gram_defect, ritz_dimension = dimension)
    end
    width = block_recurrence ? length(KrylovKit.residual(factorization)) : 1
    breakdown = KrylovKit.normres(factorization) <= sqrt(width) * residual_floor
    metrics = (;
        solver,
        recurrence = block_recurrence ? :block : :scalar,
        trial_dimension = dimension,
        appended_directions = dimension - previous_dimension,
        restarted,
        breakdown,
        recurrence_residual = KrylovKit.normres(factorization),
        gram_defect = subspace.ritz_dimension == dimension ? subspace.gram_defect : nothing,
        work_tolerance = work_tol,
        work_rank_cap = cfg.work_cap,
        max_trial_bond = maximum(rank, basis),
    )
    orbitals, subspace, metrics
end

# Davidson subspaces share the same projection and reconstruction operations.

"Canonical orthogonalization: discard small overlap eigenvalues and form W with W' * S * W = I."
function canonical_orthogonalizer(overlap, relative_cutoff, absolute_cutoff)
    metric = eigen(Hermitian((overlap + overlap') / 2))
    cutoff = max(absolute_cutoff, relative_cutoff * last(metric.values))
    independent = findall(>(cutoff), metric.values)
    metric.vectors[:, independent] * Diagonal(inv.(sqrt.(metric.values[independent])))
end

"Fit final orbitals with shared endpoint tables and bounded contraction caches."
function combine_orbitals(basis, coefficients, cfg, rec; tol, cap, label)
    evaluators = orbital_evaluators(basis, size(coefficients, 2), cfg, rec)
    recorded_map(QTT, size(coefficients, 2), cfg, rec) do i, localrec
        linear_combination(
            localrec,
            basis,
            view(coefficients, :, i);
            evaluator = orbital_evaluator(evaluators),
            tol,
            cap,
            label = "$(label)_$i",
            require_convergence = get(cfg, :adaptive_accuracy, false),
        )
    end
end

function subspace_ritz(basis, keep, project_matrices, cfg, rec; tol)
    hamiltonian, overlap = timed(() -> project_matrices(basis), rec, "subspace_projection")
    orthogonalizer =
        canonical_orthogonalizer(overlap, cfg.metric_relative_cutoff, cfg.metric_absolute_cutoff)
    size(orthogonalizer, 2) >= keep || error("The trial subspace lost occupied rank")
    projected = orthogonalizer' * hamiltonian * orthogonalizer
    spectrum = eigen(Hermitian((projected + projected') / 2))
    coefficients = orthogonalizer * spectrum.vectors[:, 1:keep]
    orbitals = combine_orbitals(
        basis,
        coefficients,
        cfg,
        rec;
        tol,
        cap = cfg.orbital_cap,
        label = "subspace_ritz",
    )
    # Fitting can change the Gram matrix; repair only if the stored orbitals need it.
    orbitals, defect = orthogonalize(orbitals, cfg, rec)
    orbitals, spectrum.values[1:keep], defect, overlap
end

function extend_subspace(basis, directions, room, cfg, rec; tol, metric = gram(basis; cfg))
    (isempty(directions) || room == 0) && return QTT[]
    norms = ttnorm.(directions)
    active = findall(>(100eps(Float64)), norms)
    directions, norms = directions[active], norms[active]
    isempty(directions) && return directions
    # Compression can spoil basis orthogonality; use the full Gram pseudoinverse.
    projection =
        pinv(
            Hermitian(metric);
            rtol = cfg.metric_relative_cutoff,
            atol = cfg.metric_absolute_cutoff,
        ) * gram(basis, directions; cfg)
    appended = KrylovKit.OrthonormalBasis{QTTVector}()
    orth = KrylovKit.ModifiedGramSchmidt2()
    norm_floor = max(tol, 100eps(Float64))
    for i in eachindex(directions)
        # Keep the projection as coefficients until an independent direction is needed.
        remainder =
            linear_expression(vcat(basis, [directions[i]]), vcat(-projection[:, i], 1) ./ norms[i])
        n = ttnorm(remainder)
        n > norm_floor || continue
        vector = QTTVector(scale(remainder, inv(n)), tol, cfg.work_cap, rec)
        # Test dependence within the new block after removing the retained space.
        vector, _ = KrylovKit.orthogonalize!!(vector, appended, orth)
        ttnorm(vector.psi)^2 > max(cfg.metric_relative_cutoff, cfg.metric_absolute_cutoff) ||
            continue
        # One fit for the accepted expression; its stored norm sets normalization.
        n = norm(vector)
        n > norm_floor || continue
        push!(appended, VI.scale!!(vector, inv(n)))
        length(appended) == room && break
    end
    reconstruct_orbital!.(appended)
end

function davidson_directions(solver, orbitals, energies, H, green, cfg, rec, potential; tol)
    residuals = recorded_map(QTT, length(orbitals), cfg, rec) do i, localrec
        h_residual(
            H,
            orbitals[i],
            energies[i],
            cfg,
            localrec;
            tol,
            cap = cfg.work_cap,
            label = "davidson_residual_$i",
            potential,
        )
    end
    norms = ttnorm.(residuals)
    selected = findall(>(max(cfg.eigen_tolerance, tol, 100eps(Float64))), norms)
    solver == :davidson && !isempty(selected) && (selected = [argmax(norms)])
    candidates = recorded_map(QTT, length(selected), cfg, rec) do j, localrec
        i = selected[j]
        # Normalize the inverse's input so its work budget is independent of residual size.
        apply_mpo(
            green,
            scale(residuals[i], inv(norms[i]));
            tol,
            cap = cfg.work_cap,
            rec = localrec,
        )
    end
    candidates, maximum(norms), norm(norms) / sqrt(length(norms))
end

"Generalized Davidson: expand with preconditioned residuals and retain the lowest Ritz orbitals."
function davidson_step(
    solver,
    orbitals,
    subspace,
    H,
    green,
    cfg,
    rec;
    work_tol,
    ritz_tol,
    potential,
    project_matrices = basis -> (h_gram(basis, H; cfg), gram(basis; cfg)),
)
    solver in (:davidson, :block_davidson) || throw(ArgumentError("Unknown Davidson method"))
    norbitals = length(orbitals)
    if subspace === nothing
        basis = orbitals
        orbitals, energies, gram_defect, metric =
            subspace_ritz(basis, norbitals, project_matrices, cfg, rec; tol = ritz_tol)
        subspace = (; basis, energies, gram_defect, metric)
    end
    limit = min(2^cfg.R, max(16, 4norbitals))
    directions, residual_max, residual_rms = timed(rec, "subspace_expansion") do
        davidson_directions(
            solver,
            orbitals,
            subspace.energies,
            H,
            green,
            cfg,
            rec,
            potential;
            tol = work_tol,
        )
    end
    # Converged orbitals do not consume expansion slots or trigger a restart.
    restarted =
        length(subspace.basis) + length(directions) > limit && length(subspace.basis) > norbitals
    basis = restarted ? orbitals : subspace.basis
    metric = restarted ? gram(basis; cfg) : subspace.metric
    room = limit - length(basis)
    appended = timed(rec, "subspace_orthogonalization") do
        extend_subspace(
            basis,
            directions,
            room,
            cfg,
            rec;
            tol = min(work_tol, cfg.orthogonalization_tolerance),
            metric,
        )
    end
    if !isempty(appended)
        basis = vcat(basis, appended)
        orbitals, energies, gram_defect, metric =
            subspace_ritz(basis, norbitals, project_matrices, cfg, rec; tol = ritz_tol)
        subspace = (; basis, energies, gram_defect, metric)
    end
    metrics = (;
        solver,
        trial_dimension = length(subspace.basis),
        appended_directions = length(appended),
        restarted = restarted && !isempty(appended),
        residual_max,
        residual_rms,
        gram_defect = subspace.gram_defect,
        work_tolerance = work_tol,
        work_rank_cap = cfg.work_cap,
        max_trial_bond = maximum(rank, subspace.basis),
    )
    orbitals, subspace, metrics
end
