# Core axes are (left bond, binary site, right bond); inner products use Euclidean weights.

# Keep this stored type stable for checkpoints; library operations use its TensorTrain view.
struct QTT
    c::Vector{Array{ComplexF64,3}}
end
QTT(c::Vector{<:Array}) = QTT([ComplexF64.(g) for g in c])
Base.length(x::QTT) = length(x.c)
tensortrain(x::QTT) = TCI.TensorTrain(x.c)
function QTT(psi::ITensorMPS.MPS)
    # TCI 0.9.19's reverse conversion needs explicit one- and two-site cases.
    length(psi) == 1 &&
        return QTT([reshape(Array(only(psi), only(ITensorMPS.siteinds(psi))), 1, 2, 1)])
    if length(psi) == 2
        sites = ITensorMPS.siteinds(psi)
        bond = ITensorMPS.linkind(psi, 1)
        return QTT([
            reshape(Array(psi[1], sites[1], bond), 1, 2, :),
            reshape(Array(psi[2], bond, sites[2]), :, 2, 1),
        ])
    end
    QTT(TCI.TensorTrain(psi).sitetensors)
end
rank(x::QTT) = length(x) == 1 ? 1 : TCI.rank(tensortrain(x))
storage(x::QTT) = sum(sizeof, x.c)
orbital_max_rank(orbitals) = maximum(rank, orbitals)
orbital_max_rank(::AbstractVector{<:AbstractVector{<:Number}}) = nothing
storage(x::AbstractVector{<:Number}) = sizeof(x)
inner(x::AbstractVector{<:Number}, y::AbstractVector{<:Number}; cfg = (;)) = dot(x, y)
ttnorm(x::AbstractVector{<:Number}; cfg = (;)) = norm(x)
difference_norm(x::AbstractVector{<:Number}, y::AbstractVector{<:Number}; kwargs...) = norm(x - y)
function gram(xs::AbstractVector{<:AbstractVector{<:Number}}; cfg = (;))
    X = reduce(hcat, xs)
    X' * X
end
gram(
    xs::AbstractVector{<:AbstractVector{<:Number}},
    ys::AbstractVector{<:AbstractVector{<:Number}};
    cfg = (;),
) = reduce(hcat, xs)' * reduce(hcat, ys)
expression_gram(xs::AbstractVector{<:AbstractVector{<:Number}}; rec = nothing) = gram(xs)
scale(x::QTT, a) = QTT(TCI.multiply(tensortrain(x), a).sitetensors)
"Contract <bra|ket> through TCI without fitting or truncating a product state."
function inner(bra::QTT, ket::QTT; cfg = (;))
    length(bra) == length(ket) || throw(DimensionMismatch("QTT inner-product lengths"))
    isempty(bra.c) && throw(ArgumentError("An inner product needs nonempty tensor trains"))
    check_compression_shapes(bra.c)
    check_compression_shapes(ket.c)
    all(size(a, 2) == size(b, 2) for (a, b) in zip(bra.c, ket.c)) ||
        throw(DimensionMismatch("QTT physical dimensions"))
    all(size(first(psi.c), 1) == size(last(psi.c), 3) == 1 for psi in (bra, ket)) ||
        throw(DimensionMismatch("QTT inner-product boundaries"))
    # A single site has no internal contraction bond in TCI 0.9.19.
    length(bra) == 1 && return dot(only(bra.c), only(ket.c))
    bra_mpo = TCI.TensorTrain([
        reshape(conj.(core), size(core, 1), 1, size(core, 2), size(core, 3)) for core in bra.c
    ])
    ket_mpo = TCI.TensorTrain([
        reshape(core, size(core, 1), size(core, 2), 1, size(core, 3)) for core in ket.c
    ])
    TCI.evaluate(TCI.Contraction(bra_mpo, ket_mpo), fill((1, 1), length(bra)))
end
ttnorm(x::QTT; cfg = (;)) = sqrt(max(0, real(inner(x, x; cfg))))

"Convert a shared orbital basis and its operators once per projection."
function mpo_basis(orbitals, operators)
    sites = [ITensors.Index(2, "n=$k") for k = 1:length(first(orbitals))]
    kets = [ITensorMPS.MPS(tensortrain(psi); sites) for psi in orbitals]
    operator_sites = [[s', s] for s in sites]
    matrices = [ITensorMPS.MPO(TCI.TensorTrain(w); sites = operator_sites) for w in operators]
    ITensors.prime.(kets), kets, matrices
end

"Contract <x|W|y> without truncating or constructing W*y."
function inner_mpo(x::QTT, w, y::QTT)
    length(x) == length(y) == length(w) || throw(DimensionMismatch("MPO expectation lengths"))
    # Random index IDs must not advance the TCI sampling stream.
    with_sampling_seed(0) do
        bras, kets, operators = mpo_basis(x === y ? (x,) : (x, y), (w,))
        ITensorMPS.inner(first(bras), only(operators), last(kets); make_inds_match = false)
    end
end
normalize(x::QTT; cfg = (;)) = scale(x, 1 / ttnorm(x; cfg))
function validate_truncation(tol, cap)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("Invalid QTT tolerance"))
    cap >= 1 || throw(ArgumentError("QTT rank cap must be positive"))
    nothing
end
function rounded(x::QTT; tol = 1e-11, cap = 64)
    validate_truncation(tol, cap)
    tt = with_svd_context(;
        operation = "rounded",
        original_cores = x.c,
        tolerance = tol,
        rank_cap = cap,
    ) do
        check_compression_shapes(x.c)
        # TCI's nonmutating SVD replaces every core for multi-site trains.
        result = TCI.TensorTrain(length(x) == 1 ? [copy(only(x.c))] : copy(x.c))
        compress_with_recovery!(result; tolerance = Float64(tol), maxbonddim = cap)
        result
    end
    QTT(tt.sitetensors)
end

"Pointwise addition uses recorded TCI when rec is supplied, otherwise library TT addition."
function add(
    x::QTT,
    y::QTT;
    a = 1,
    b = 1,
    tol = 1e-11,
    cap = 64,
    rec = nothing,
    label = "add",
    require_convergence = false,
)
    validate_truncation(tol, cap)
    length(x) == length(y) || throw(DimensionMismatch("QTT addition lengths"))
    if rec !== nothing
        return linear_combination(rec, (x, y), (a, b); tol, cap, label, require_convergence)
    end
    # TCI 0.9.19's block addition assumes distinct first and last sites.
    length(x) == 1 && return QTT([a .* only(x.c) .+ b .* only(y.c)])
    try
        result = TCI.add(
            tensortrain(x),
            tensortrain(y);
            factorlhs = a,
            factorrhs = b,
            tolerance = Float64(tol),
            maxbonddim = cap,
        )
        all(core -> all(isfinite, core), result.sitetensors) && return QTT(result.sitetensors)
    catch error
        retryable_svd_error(error) || rethrow()
    end
    # Reuse the amplitude-space sum when library addition fails numerically.
    reconstruct_expression(linear_expression((x, y), (a, b)); tol, cap)
end
function constant_tt(R, value = 1)
    c = [ones(ComplexF64, 1, 2, 1) for k = 1:R]
    c[1] .*= value
    QTT(c)
end
function phase_tt(x::QTT, k::Int)
    N = 2^length(x)
    c = [copy(g) for g in x.c]
    for j = 1:length(x)
        c[j][:, 2, :] .*= cis(2pi * k * 2^(j - 1) / N)
    end
    QTT(c)
end
"Convert each orbital once before contracting a block of pairwise overlaps."
function mps_basis(orbitals, sites)
    [
        begin
            isempty(psi.c) && throw(ArgumentError("An inner product needs nonempty tensor trains"))
            length(psi) == length(sites) || throw(DimensionMismatch("QTT inner-product lengths"))
            check_compression_shapes(psi.c)
            all(size(g, 2) == ITensors.dim(s) for (g, s) in zip(psi.c, sites)) ||
                throw(DimensionMismatch("QTT physical dimensions"))
            size(first(psi.c), 1) == size(last(psi.c), 3) == 1 ||
                throw(DimensionMismatch("QTT inner-product boundaries"))
            ITensorMPS.MPS(tensortrain(psi); sites)
        end for psi in orbitals
    ]
end
function gram(xs; cfg = (;))
    g = zeros(ComplexF64, length(xs), length(xs))
    isempty(xs) && return g
    # Conversion and the library's internal link IDs must not consume TCI randomness.
    with_sampling_seed(0) do
        sites = [ITensors.Index(size(core, 2), "n=$k") for (k, core) in enumerate(first(xs).c)]
        kets = mps_basis(xs, sites)
        foreach_independent(length(xs), cfg) do j
            with_sampling_seed(j) do
                for i = firstindex(xs):j
                    value = ITensorMPS.inner(kets[i], kets[j]; make_inds_match = false)
                    g[i, j] = value
                    i == j || (g[j, i] = conj(value))
                end
            end
        end
    end
    g
end
function gram(xs, ys; cfg = (;))
    g = zeros(ComplexF64, length(xs), length(ys))
    (isempty(xs) || isempty(ys)) && return g
    with_sampling_seed(0) do
        sites = [ITensors.Index(size(core, 2), "n=$k") for (k, core) in enumerate(first(xs).c)]
        left = mps_basis(xs, sites)
        right = mps_basis(ys, sites)
        foreach_independent(length(ys), cfg) do j
            with_sampling_seed(j) do
                for i in eachindex(xs)
                    g[i, j] = ITensorMPS.inner(left[i], right[j]; make_inds_match = false)
                end
            end
        end
    end
    g
end

# Evaluate integer-labelled samples; the first core holds the least-significant bit.

mutable struct Evaluator
    x::QTT
    buffer::Vector{ComplexF64}
    next_buffer::Vector{ComplexF64}
    left::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    prefix::Int
    suffix::Int
end
"Choose endpoint tables from actual bond sizes and a block-wide byte budget."
function endpoint_plan(xs, max_bytes)
    R = length(first(xs))
    all(x -> length(x) == R, xs) || throw(DimensionMismatch("Endpoint grids"))
    buffer_bytes = 32sum(x -> max(rank(x), 1), xs)
    max_bytes >= 32length(xs) + buffer_bytes ||
        throw(ArgumentError("Endpoint budget is too small for contraction buffers"))
    ranks = [sum(p == 0 ? 1 : size(x.c[p], 3) for x in xs) for p = 0:R]
    costs = [sum(size(x.c[k], 1) * size(x.c[k], 3) for x in xs) for k = 1:R]
    limit = min(R, floor(Int, log2((max_bytes - buffer_bytes) / 16)))
    left_extra, right_extra = zeros(Int, limit + 1), zeros(Int, limit + 1)
    for x in xs
        left_sizes = [16 * (1 << k) * (k == 0 ? 1 : size(x.c[k], 3)) for k = 0:limit]
        right_sizes = [16 * (1 << k) * (k == 0 ? 1 : size(x.c[R - k + 1], 1)) for k = 0:limit]
        left_peak, right_peak = 0, 0
        for k = 1:limit
            left_peak = max(left_peak, left_sizes[k] + left_sizes[k + 1])
            right_peak = max(right_peak, right_sizes[k] + right_sizes[k + 1])
            left_extra[k + 1] = max(left_extra[k + 1], left_peak - left_sizes[k + 1])
            right_extra[k + 1] = max(right_extra[k + 1], right_peak - right_sizes[k + 1])
        end
    end
    best = (
        prefix = 0,
        suffix = 0,
        bytes = 32length(xs),
        buffer_bytes,
        peak_bytes = 32length(xs) + buffer_bytes,
        cost = sum(costs),
    )
    for p = 0:limit, s = 0:min(limit, R - p)
        R > 1 && (p == R || s == R) && continue
        bytes = 16 * ((1 << p) * ranks[p + 1] + (1 << s) * ranks[R - s + 1])
        peak_bytes = bytes + buffer_bytes + max(left_extra[p + 1], right_extra[s + 1])
        peak_bytes <= max_bytes || continue
        # Once endpoints meet, choose the smallest tables instead of a near-dense lookup.
        cost = sum(view(costs, (p + 1):(R - s)); init = 0)
        (cost < best.cost || (cost == best.cost && bytes < best.bytes)) &&
            (best = (; prefix = p, suffix = s, bytes, buffer_bytes, peak_bytes, cost))
    end
    best
end

"Build each endpoint by matrix products without a second dictionary of partial tables."
function endpoint_table(x, count, left)
    table = ones(ComplexF64, 1, 1)
    for k = 1:count
        core = x.c[left ? k : length(x) - k + 1]
        n = size(table, 2)
        next = Matrix{ComplexF64}(undef, size(core, left ? 3 : 1), 2n)
        for digit = 1:2
            columns = left ? (((digit - 1) * n + 1):(digit * n)) : (digit:2:(2n))
            if left
                mul!(view(next, :, columns), transpose(view(core, :, digit, :)), table)
            else
                mul!(view(next, :, columns), view(core, :, digit, :), table)
            end
        end
        table = next
    end
    table
end

function Evaluator(x::QTT; max_bytes = 8 * 1024^2, plan = endpoint_plan((x,), max_bytes))
    p, s = plan.prefix, plan.suffix
    l = endpoint_table(x, p, true)
    r = endpoint_table(x, s, false)
    Evaluator(x, zeros(ComplexF64, max(rank(x), 1)), zeros(ComplexF64, max(rank(x), 1)), l, r, p, s)
end
function (e::Evaluator)(j::Int)
    0 <= j < (1 << length(e.x)) || throw(BoundsError(e.x, j))
    li = (j & ((1 << e.prefix) - 1)) + 1
    ri = (j >> (length(e.x) - e.suffix)) + 1
    if e.prefix + e.suffix == length(e.x)
        return transpose(view(e.left, :, li)) * view(e.right, :, ri)
    end
    copyto!(e.buffer, 1, e.left, 1 + (li - 1) * size(e.left, 1), size(e.left, 1))
    for site = (e.prefix + 1):(length(e.x) - e.suffix)
        core = e.x.c[site]
        digit = ((j >> (site - 1)) & 1) + 1
        mul!(
            view(e.next_buffer, 1:size(core, 3)),
            transpose(view(core, :, digit, :)),
            view(e.buffer, 1:size(core, 1)),
        )
        e.buffer, e.next_buffer = e.next_buffer, e.buffer
    end
    transpose(view(e.buffer, 1:size(e.right, 1))) * view(e.right, :, ri)
end
bits(j, R) = [((j >> (k - 1)) & 1) + 1 for k = 1:R]
indexof(v) = sum(((v[k] - 1) << (k - 1) for k = 1:length(v)); init = 0)

# A linear expression stores coefficients and QTT terms until reconstruction is required.

struct QTTExpression
    terms::Vector{QTT}
    coefficients::Vector{ComplexF64}
    R::Int
end
Base.length(x::QTTExpression) = x.R

function linear_expression(xs, weights)
    if !isempty(xs) && first(xs) isa AbstractVector{<:Number}
        return reduce(hcat, xs) * collect(weights)
    end
    isempty(xs) && throw(ArgumentError("A linear expression needs a grid"))
    length(xs) == length(weights) || throw(DimensionMismatch("Expression coefficients"))
    R = length(first(xs))
    terms = QTT[]
    coefficients = ComplexF64[]
    positions = IdDict{Vector{Array{ComplexF64,3}},Int}()
    function append_term(x::QTT, c)
        if haskey(positions, x.c)
            coefficients[positions[x.c]] += c
        else
            push!(terms, x)
            push!(coefficients, c)
            positions[x.c] = length(terms)
        end
    end
    for (x, c) in zip(xs, weights)
        length(x) == R || throw(DimensionMismatch("Expression grid lengths"))
        isfinite(c) || throw(ArgumentError("Nonfinite expression coefficient"))
        if x isa QTTExpression
            for (term, weight) in zip(x.terms, x.coefficients)
                append_term(term, c * weight)
            end
        else
            append_term(x, c)
        end
    end
    all(isfinite, coefficients) || throw(ArgumentError("Nonfinite combined coefficients"))
    active = findall(!iszero, coefficients)
    QTTExpression(terms[active], coefficients[active], R)
end
scale(x::QTTExpression, a) = linear_expression((x,), (a,))

# Compensate the final scalar/amplitude sums, where cancellation is intended.
function compensated_sum(values)
    total = zero(ComplexF64)
    correction = zero(ComplexF64)
    for value in values
        adjusted = value - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    end
    total
end

inner(x::QTTExpression, y::QTT; cfg = (;)) =
    compensated_sum(conj(c) * inner(t, y; cfg) for (t, c) in zip(x.terms, x.coefficients))
inner(x::QTT, y::QTTExpression; cfg = (;)) = conj(inner(y, x; cfg))
inner(x::QTTExpression, y::QTTExpression; cfg = (;)) =
    compensated_sum(c * inner(x, t; cfg) for (t, c) in zip(y.terms, y.coefficients))

struct ExpressionEvaluator
    evaluators::Vector{Evaluator}
    coefficients::Vector{ComplexF64}
    R::Int
end
Evaluator(x::QTTExpression; max_bytes = 8 * 1024^2) = ExpressionEvaluator(
    [Evaluator(t; max_bytes = max_bytes ÷ max(1, length(x.terms))) for t in x.terms],
    x.coefficients,
    x.R,
)
function (e::ExpressionEvaluator)(j::Int)
    0 <= j < (1 << e.R) || throw(BoundsError(e, j))
    compensated_sum(c * f(j) for (c, f) in zip(e.coefficients, e.evaluators))
end

function expression_components(xs)
    isempty(xs) && throw(ArgumentError("Empty expression collection"))
    R = length(first(xs))
    terms = QTT[]
    positions = IdDict{Vector{Array{ComplexF64,3}},Int}()
    for x in xs
        length(x) == R || throw(DimensionMismatch("Expression grid lengths"))
        for term in x.terms
            if !haskey(positions, term.c)
                push!(terms, term)
                positions[term.c] = length(terms)
            end
        end
    end
    weights = zeros(ComplexF64, length(terms), length(xs))
    for (j, x) in enumerate(xs), (term, c) in zip(x.terms, x.coefficients)
        weights[positions[term.c], j] += c
    end
    terms, weights
end

expression_stage(f, rec, label) = rec === nothing ? f() : timed(f, rec, label)

"Stream an untruncated common QR basis; retain its cores only for final reconstruction."
function expression_qr(terms, weights; keep_cores = false)
    isempty(terms) && throw(ArgumentError("QR needs at least one nonzero term"))
    with_svd_context(;
        operation = "linear_expression_qr",
        original_cores = [x.c for x in terms],
        expression_coefficients = weights,
    ) do
        for x in terms
            check_compression_shapes(x.c)
            !isempty(x.c) &&
                size(first(x.c), 1) == size(last(x.c), 3) == 1 &&
                all(g -> size(g, 2) == 2, x.c) || throw(
                DimensionMismatch("Expression QR requires binary QTT cores and scalar boundaries"),
            )
        end
        require_finite(weights, "nonfinite_expression_coefficients")
        right_factors = [ones(ComplexF64, 1, 1) for _ in terms]
        cores = Array{ComplexF64,3}[]
        R = length(first(terms))
        p = 1
        for k = 1:R
            widths = [size(x.c[k], 3) for x in terms]
            A = zeros(ComplexF64, 2p, sum(widths))
            offset = 0
            for (i, x) in enumerate(terms)
                columns = (offset + 1):(offset + widths[i])
                for s = 1:2
                    mul!(
                        view(A, ((s - 1) * p + 1):(s * p), columns),
                        right_factors[i],
                        view(x.c[k], :, s, :),
                    )
                end
                offset += widths[i]
            end
            require_finite(A, "nonfinite_expression_qr_input"; details = (; site = k))
            if k == R
                # Each final right bond is one. Combine in amplitude space,
                # not by subtracting almost equal squared norms.
                C = [
                    compensated_sum(A[i, t] * weights[t, j] for t in eachindex(terms)) for
                    i in axes(A, 1), j in axes(weights, 2)
                ]
                require_finite(C, "nonfinite_expression_amplitudes")
                return C, cores
            end
            F = try
                qr(A)
            catch e
                save_svd_event(
                    A,
                    "expression_qr_exception";
                    details = (; site = k, exception = sprint(showerror, e)),
                )
                rethrow()
            end
            triangular = Matrix(F.R)
            require_finite(triangular, "nonfinite_expression_qr_output"; details = (; site = k))
            q = size(triangular, 1)
            if keep_cores
                # Applying Q to the thin identity avoids a full square Q.
                basis = F.Q * Matrix{ComplexF64}(I, size(A, 1), q)
                require_finite(basis, "nonfinite_expression_q_basis"; details = (; site = k))
                push!(cores, reshape(basis, p, 2, q))
            end
            offset = 0
            for i in eachindex(terms)
                right_factors[i] = triangular[:, (offset + 1):(offset + widths[i])]
                offset += widths[i]
            end
            p = q
        end
        error("Empty QTT grid")
    end
end

"Gram from original TCI inner products, with QR fallback for cancellation or near dependence."
function expression_gram(xs; rec = nothing)
    terms, W = expression_components(xs)
    m = length(xs)
    isempty(terms) && return zeros(ComplexF64, m, m)
    K = expression_stage(rec, "density_gram_core_inner") do
        result = gram(terms; cfg = rec === nothing ? (;) : rec.options)
        require_finite(result, "nonfinite_expression_core_gram")
        result
    end
    G = W' * K * W
    magnitudes = abs.(W)
    scales = diag(transpose(magnitudes) * abs.(K) * magnitudes)
    require_finite(G, "nonfinite_expression_gram")
    require_finite(scales, "nonfinite_expression_gram_scale")
    # Small norms need an amplitude-space QR, not subtraction of squared norms.
    fallback = any(1:m) do i
        real(G[i, i]) <= sqrt(eps(Float64)) * scales[i] ||
            abs(imag(G[i, i])) > 64eps(Float64) * scales[i]
    end
    G = Matrix(Hermitian(G, :U))
    if m > 1
        fallback |= eigmin(Hermitian(G)) <= sqrt(eps(Float64)) * maximum(scales)
    end
    if fallback
        return expression_stage(rec, "density_gram_qr_fallback") do
            C, _ = expression_qr(terms, W)
            result = C' * C
            result[diagind(result)] .= vec(sum(abs2, C; dims = 1))
            require_finite(result, "nonfinite_expression_qr_gram")
            result
        end
    end
    G
end

ttnorm(x::QTTExpression) = sqrt(max(0, real(only(expression_gram([x])))))

function difference_norm(x::QTT, y::QTT; rec = nothing, label = "density_difference_norm")
    expression_stage(rec, label) do
        residual = linear_expression((x, y), (1, -1))
        squared = real(only(expression_gram([residual]; rec)))
        squared >= 0 || error("Negative residual norm after stable contraction")
        sqrt(squared)
    end
end

"Reconstruct the linear combination, then round it with numerical failure recovery."
function reconstruct_expression(x::QTTExpression; tol, cap, absolute_tolerance = nothing)
    validate_truncation(tol, cap)
    isempty(x.terms) && return constant_tt(x.R, 0)
    if length(x.terms) == 1
        value = scale(only(x.terms), only(x.coefficients))
        return absolute_tolerance === nothing ? rounded(value; tol, cap) :
               rounded_absolute(value, absolute_tolerance; cap)
    end
    C, cores = expression_qr(x.terms, reshape(x.coefficients, :, 1); keep_cores = true)
    push!(cores, reshape(C, size(C, 1) ÷ 2, 2, 1))
    value = QTT(cores)
    absolute_tolerance === nothing ? rounded(value; tol, cap) :
    rounded_absolute(value, absolute_tolerance; cap, amplitude_norm = norm(C))
end

"Round only when the measured amplitude error satisfies its absolute budget."
function rounded_absolute(x, tolerance; cap, amplitude_norm = ttnorm(x))
    isfinite(tolerance) && tolerance > 0 ||
        throw(ArgumentError("Invalid absolute amplitude tolerance"))
    amplitude_norm <= tolerance && return constant_tt(length(x), 0.0)
    relative = tolerance / (amplitude_norm * max(1, length(x) - 1))
    for attempt = 1:4
        value = rounded(x; tol = relative, cap)
        difference_norm(value, x) <= tolerance && return value
        relative /= 10
    end
    error("Tensor rounding could not meet the absolute amplitude budget $tolerance")
end

# MPO construction and action; axes are left bond, output site, input site, right bond.

"Circulant C[i,j]=kernel[(i-j) mod N] using a two-state binary borrow automaton."
function circulant(kernel::QTT)
    out = Array{ComplexF64,4}[]
    d = length(kernel)
    for (k, g) in enumerate(kernel.c)
        rl, _, rr = size(g)
        nl = k == 1 ? 1 : 2
        nr = k == d ? 1 : 2
        w = zeros(ComplexF64, rl * nl, 2, 2, rr * nr)
        for bin = 0:(nl - 1), i = 0:1, j = 0:1
            z = i - j - bin
            s = mod(z, 2) + 1
            bout = z < 0 ? 1 : 0
            left = (1:rl) .+ rl * bin
            right = (1:rr) .+ rr * (k == d ? 0 : bout)
            @views w[left, i + 1, j + 1, right] .+= g[:, s, :]
        end
        push!(out, w)
    end
    out
end
function diagonal(x::QTT)
    [
        begin
            w = zeros(ComplexF64, size(g, 1), 2, 2, size(g, 3))
            w[:, 1, 1, :] = g[:, 1, :]
            w[:, 2, 2, :] = g[:, 2, :]
            w
        end for g in x.c
    ]
end

"Unit shift with LSB-first carry states; periodic overflow wraps around."
function shift_mpo(R::Int, step::Int = 1; boundary = :periodic)
    1 <= R <= 60 || throw(ArgumentError("Shift needs 1:60 binary sites"))
    abs(step) == 1 || throw(ArgumentError("Only unit shifts are supported"))
    boundary in (:periodic, :open) || throw(ArgumentError("Unknown shift boundary"))
    cores = Array{ComplexF64,4}[]
    for k = 1:R
        w = zeros(ComplexF64, k == 1 ? 1 : 2, 2, 2, k == R ? 1 : 2)
        for carry in (k == 1 ? (1:1) : (0:1)), input = 0:1
            total = input + carry
            output, next = mod(total, 2), total ÷ 2
            k == R && boundary == :open && next != 0 && continue
            w[k == 1 ? 1 : carry + 1, output + 1, input + 1, k == R ? 1 : next + 1] = 1
        end
        push!(cores, step == 1 ? w : permutedims(w, (1, 3, 2, 4)))
    end
    cores
end
"Apply an MPO with library-owned canonicalization and rank control."
function apply_mpo(w, x::QTT; tol = 1e-11, cap = 64, algorithm = :zipup, rec = nothing)
    validate_truncation(tol, cap)
    length(w) == length(x) || throw(DimensionMismatch("MPO action lengths"))
    if algorithm === :TCI
        product = TCI.contract(
            TCI.TensorTrain(w),
            tensortrain(x);
            algorithm,
            tolerance = Float64(tol),
            maxbonddim = cap,
        )
        return QTT(product.sitetensors)
    end
    algorithm in (:zipup, :naive, :fit) || throw(ArgumentError("Unknown MPO algorithm: $algorithm"))
    result = with_sampling_seed(0) do
        psi = ITensorMPS.MPS(tensortrain(x))
        sites = ITensorMPS.siteinds(psi)
        operator = ITensorMPS.MPO(TCI.TensorTrain(w); sites = [[s', s] for s in sites])
        exact =
            iszero(tol) && all(k -> size(w[k], 4) * size(x.c[k], 3) <= cap, 1:(length(x) - 1))
        product = if exact
            ITensorMPS.apply(operator, psi; alg = "naive", truncate = false)
        elseif algorithm === :naive
            ITensorMPS.apply(operator, psi; alg = "naive", cutoff = tol^2, maxdim = cap)
        elseif algorithm === :fit
            ITensorMPS.apply(
                operator, psi; alg = "fit", nsweeps = 2, nsite = 2,
                normalize = false, cutoff = tol^2, maxdim = cap, outputlevel = 0,
            )
        else
            # ITensor cutoffs bound discarded squared weight, not amplitude error.
            # zip-up canonicalizes both operands and owns the final truncation.
            ITensorMPS.apply(
                operator,
                psi;
                alg = "zipup",
                cutoff = (tol / length(x))^2,
                maxdim = cap,
                truncate_kwargs = (; cutoff = tol^2, maxdim = cap),
            )
        end
        QTT(product)
    end
    foreach(core -> require_finite(core, "nonfinite_mpo_result"), result.c)
    result
end
function delta_tt(R, j)
    QTT([reshape(ComplexF64[s == ((j >> (k - 1)) & 1) + 1 for s = 1:2], 1, 2, 1) for k = 1:R])
end
function green_kernel(R, h, mu)
    1 <= R <= 60 || throw(ArgumentError("Green needs 1:60 binary sites"))
    isfinite(h) && h > 0 && isfinite(mu) && !iszero(mu) && real(mu) >= 0 || throw(
        ArgumentError(
            "Green needs positive spacing and a nonzero shift in the closed right half-plane",
        ),
    )
    gamma = 2asinh(h * sqrt(mu / 2))
    N = 2^R
    # Two rank-one exponentials; never form a vector of length N.
    p = QTT([reshape(ComplexF64[1, exp(-gamma * 2^(k - 1))], 1, 2, 1) for k = 1:R])
    q = QTT([reshape(ComplexF64[exp(-gamma * 2^(k - 1)), 1], 1, 2, 1) for k = 1:R])
    kern = add(p, scale(q, exp(-gamma)); tol = 1e-14, cap = 2)
    scale(kern, h^2 / (sinh(gamma) * (-expm1(-gamma * N))))
end
green_mpo(R, h, mu) = circulant(green_kernel(R, h, mu))

"Periodic FD2 spectral filter with gain 1 / (1 + (lambda / cutoff)^2)."
function lowpass_mpo(R, h, cutoff)
    isfinite(cutoff) && cutoff > 0 ||
        throw(ArgumentError("Low-pass cutoff must be finite and positive"))
    scaled_h = h * sqrt(cutoff)
    isfinite(scaled_h^2) && scaled_h^2 > 0 ||
        throw(ArgumentError("Invalid scaled low-pass spacing"))
    # With spacing h*sqrt(cutoff), the Green kernel inverts T/cutoff + iI.
    kernel = scale(green_kernel(R, scaled_h, im), im)
    # Carry real and imaginary amplitudes to take the real part of the full product.
    cores = map(eachindex(kernel.c)) do k
        re, ii = real.(kernel.c[k]), imag.(kernel.c[k])
        if R == 1
            ComplexF64.(re)
        elseif k == 1
            ComplexF64.(cat(re, -ii; dims = 3))
        elseif k == R
            ComplexF64.(cat(re, ii; dims = 1))
        else
            ComplexF64.(cat(cat(re, -ii; dims = 3), cat(ii, re; dims = 3); dims = 1))
        end
    end
    circulant(QTT(cores))
end

# Interpolate neighboring periodic samples to evaluate a QTT on a finer grid.

"Periodic degree-five interpolation of normalized grid amplitudes, using six TT point queries."
struct PeriodicRefinement{E}
    evaluate::E
    coarse_size::Int
    ratio::Int
    nodes::Vector{Int}
    weights::Matrix{Float64}
end
function PeriodicRefinement(x::AbstractVector{<:Number}, target_R::Int; degree = 5, kwargs...)
    q, remainder = divrem(1 << target_R, length(x))
    q > 1 && remainder == 0 || throw(ArgumentError("Refinement requires a finer binary grid"))
    nodes = collect((-(degree ÷ 2)):(degree ÷ 2 + 1))
    weights = [
        prod((s / q - b) / (a - b) for b in nodes if b != a) / sqrt(q) for a in nodes, s = 0:(q - 1)
    ]
    PeriodicRefinement(j -> x[j + 1], length(x), q, nodes, weights)
end
function PeriodicRefinement(x::QTT, target_R::Int; degree = 5, max_bytes = 8 * 1024^2)
    target_R > length(x) || throw(ArgumentError("Refinement requires a finer grid"))
    q = 1 << (target_R - length(x))
    nodes = collect((-(degree ÷ 2)):(degree ÷ 2 + 1))
    weights = [
        prod((s / q - b) / (a - b) for b in nodes if b != a) / sqrt(q) for a in nodes, s = 0:(q - 1)
    ]
    PeriodicRefinement(Evaluator(x; max_bytes), 1 << length(x), q, nodes, weights)
end
function (p::PeriodicRefinement)(j::Int)
    coarse, offset = divrem(j, p.ratio)
    offset == 0 && return p.evaluate(mod(coarse, p.coarse_size)) / sqrt(p.ratio)
    sum(
        p.weights[k, offset + 1] * p.evaluate(mod(coarse + p.nodes[k], p.coarse_size)) for
        k in eachindex(p.nodes)
    )
end

# SVD failure records and recovery; diagnostic context belongs to the current task.

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

"Compress with TCI; recover from the preserved cores through ITensorMPS."
function compress_with_recovery!(
    tt;
    tolerance = 1e-12,
    maxbonddim = typemax(Int),
    compress = TCI.compress!,
)
    check_compression_shapes(tt.sitetensors)
    for (site, core) in enumerate(tt.sitetensors)
        with_svd_context(; site, direction = "input") do
            require_finite(core, "nonfinite_compression_input")
        end
    end
    # TCI replaces cores without mutating the originals, so a shallow copy is enough.
    original_cores = copy(tt.sitetensors)
    failure = try
        compress(tt, :SVD; tolerance, maxbonddim)
        all(core -> all(isfinite, core), tt.sitetensors) && return nothing
        "TCI compression returned nonfinite cores"
    catch exception
        intermediate_nonfinite =
            exception isa ArgumentError &&
            occursin("matrix contains Infs or NaNs", sprint(showerror, exception))
        retryable_svd_error(exception) || intermediate_nonfinite || rethrow()
        sprint(showerror, exception)
    end
    tt.sitetensors .= original_cores
    event = save_svd_event(
        first(original_cores),
        "tci_compression_failed";
        details = (; original_cores, failure),
    )
    try
        recovered = with_sampling_seed(0) do
            psi = ITensorMPS.MPS(TCI.TensorTrain(original_cores))
            # ITensor's SVD backend retries failed factorizations with QR iteration.
            # Its cutoff is a discarded squared weight; our tolerance is an amplitude.
            ITensorMPS.truncate!(psi; cutoff = tolerance^2, maxdim = maxbonddim)
            QTT(psi)
        end
        all(core -> all(isfinite, core), recovered.c) ||
            error("ITensorMPS compression returned nonfinite cores")
        tt.sitetensors .= recovered.c
        finish_svd_event(event, "recovered_with_itensormps")
    catch exception
        finish_svd_event(event, "recovery_failed"; error = sprint(showerror, exception))
        rethrow()
    end
    nothing
end
