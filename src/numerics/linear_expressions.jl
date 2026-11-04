# Flat linear expressions retain QTT cores until a consumer needs new cores.

struct QTTExpression
    terms::Vector{QTT}
    coefficients::Vector{ComplexF64}
    R::Int
end
Base.length(x::QTTExpression) = x.R

function linear_expression(xs, weights)
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

struct ExpressionEvaluator
    evals::Vector{Evaluator}
    coefficients::Vector{ComplexF64}
    R::Int
end
Evaluator(x::QTTExpression; cachebits = 10) =
    ExpressionEvaluator([Evaluator(t; cachebits) for t in x.terms], x.coefficients, x.R)
function (e::ExpressionEvaluator)(j::Int)
    0 <= j < (1 << e.R) || throw(BoundsError(e, j))
    compensated_sum(c * f(j) for (c, f) in zip(e.coefficients, e.evals))
end

function expression_components(xs)
    isempty(xs) && throw(ArgumentError("Empty expression collection"))
    R = length(first(xs))
    leaves = QTT[]
    positions = IdDict{Vector{Array{ComplexF64,3}},Int}()
    for x in xs
        length(x) == R || throw(DimensionMismatch("Expression grid lengths"))
        for term in x.terms
            if !haskey(positions, term.c)
                push!(leaves, term)
                positions[term.c] = length(leaves)
            end
        end
    end
    weights = zeros(ComplexF64, length(leaves), length(xs))
    for (j, x) in enumerate(xs), (term, c) in zip(x.terms, x.coefficients)
        weights[positions[term.c], j] += c
    end
    leaves, weights
end

expression_stage(f, rec, label) = rec === nothing ? f() : timed(f, rec, label)

"Stream an untruncated common QR basis; retain its cores only for final materialization."
function expression_qr(leaves, weights; keep_cores = false)
    isempty(leaves) && throw(ArgumentError("QR needs at least one nonzero term"))
    with_svd_context(;
        operation = "linear_expression_qr",
        original_cores = [x.c for x in leaves],
        expression_coefficients = weights,
    ) do
        for x in leaves
            check_compression_shapes(x.c)
            !isempty(x.c) &&
                size(first(x.c), 1) == size(last(x.c), 3) == 1 &&
                all(g -> size(g, 2) == 2, x.c) || throw(
                DimensionMismatch("Expression QR requires binary QTT cores and scalar boundaries"),
            )
        end
        require_finite(weights, "nonfinite_expression_coefficients")
        carry = [ones(ComplexF64, 1, 1) for _ in leaves]
        cores = Array{ComplexF64,3}[]
        R = length(first(leaves))
        p = 1
        for k = 1:R
            widths = [size(x.c[k], 3) for x in leaves]
            A = zeros(ComplexF64, 2p, sum(widths))
            offset = 0
            for (i, x) in enumerate(leaves)
                columns = (offset + 1):(offset + widths[i])
                for s = 1:2
                    mul!(
                        view(A, ((s - 1) * p + 1):(s * p), columns),
                        carry[i],
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
                    compensated_sum(A[i, t] * weights[t, j] for t in eachindex(leaves)) for
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
            for i in eachindex(leaves)
                carry[i] = triangular[:, (offset + 1):(offset + widths[i])]
                offset += widths[i]
            end
            p = q
        end
        error("Empty QTT grid")
    end
end

"Gram from original TCI inner products, with QR fallback for cancellation or near dependence."
function expression_gram(xs; rec = nothing)
    leaves, W = expression_components(xs)
    m = length(xs)
    isempty(leaves) && return zeros(ComplexF64, m, m)
    K = expression_stage(rec, "density_gram_core_inner") do
        result = gram(leaves; cfg = rec === nothing ? (;) : rec.options)
        require_finite(result, "nonfinite_expression_core_gram")
        result
    end
    G = zeros(ComplexF64, m, m)
    scales = zeros(m)
    fallback = false
    for j = 1:m, i = 1:j
        value = compensated_sum(
            conj(W[a, i]) * K[a, b] * W[b, j] for a in eachindex(leaves), b in eachindex(leaves)
        )
        G[i, j] = value
        G[j, i] = conj(value)
        if i == j
            scales[i] = sum(
                abs(conj(W[a, i]) * K[a, b] * W[b, i]) for a in eachindex(leaves),
                b in eachindex(leaves)
            )
            # This is a conservative switch, not an error bound. Recompute
            # small or negative norms instead of clipping them to zero.
            fallback |=
                real(value) <= sqrt(eps(Float64)) * scales[i] ||
                abs(imag(value)) > 64eps(Float64) * scales[i]
            G[i, i] = real(value)
        end
    end
    require_finite(G, "nonfinite_expression_gram")
    require_finite(scales, "nonfinite_expression_gram_scale")
    if m > 1
        fallback |= eigmin(Hermitian(G)) <= sqrt(eps(Float64)) * maximum(scales)
    end
    if fallback
        return expression_stage(rec, "density_gram_qr_fallback") do
            C, _ = expression_qr(leaves, W)
            result = C' * C
            for j = 1:m
                result[j, j] = sum(abs2, view(C, :, j))
            end
            require_finite(result, "nonfinite_expression_qr_gram")
            result
        end
    end
    G
end

function difference_norm(x::QTT, y::QTT; rec = nothing, label = "density_difference_norm")
    expression_stage(rec, label) do
        residual = linear_expression((x, y), (1, -1))
        squared = real(only(expression_gram([residual]; rec)))
        squared >= 0 || error("Negative residual norm after stable contraction")
        sqrt(squared)
    end
end

"Build just the final linear combination, then apply the existing guarded SVD rounding."
function materialize_expression(x::QTTExpression; tol, cap)
    algebra_options(tol, cap)
    isempty(x.terms) && return constant_tt(x.R, 0)
    length(x.terms) == 1 && return rounded(scale(only(x.terms), only(x.coefficients)); tol, cap)
    C, cores = expression_qr(x.terms, reshape(x.coefficients, :, 1); keep_cores = true)
    push!(cores, reshape(C, size(C, 1) ÷ 2, 2, 1))
    rounded(QTT(cores); tol, cap)
end
