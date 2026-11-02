# Core axes are (left bond, binary site, right bond); inner products use Euclidean weights.

struct QTT
    c::Vector{Array{ComplexF64,3}}
end
QTT(c::Vector{<:Array}) = QTT([ComplexF64.(g) for g in c])
Base.length(x::QTT) = length(x.c)
tensortrain(x::QTT) = TCI.TensorTrain(x.c)
rank(x::QTT) = length(x) == 1 ? 1 : TCI.rank(tensortrain(x))
storage(x::QTT) = sum(sizeof, x.c)
scale(x::QTT, a) = QTT(TCI.multiply(tensortrain(x), a).sitetensors)
function inner(x::QTT, y::QTT)
    length(x) == length(y) || throw(DimensionMismatch("QTT inner-product lengths"))
    # Treat the bra and ket as 1-by-N and N-by-1 tensor-train operators.
    # Evaluating their sole output entry contracts the inner product in TCI
    # without constructing a pointwise product or fitting another tensor train.
    bra = TCI.TensorTrain([reshape(conj.(g), size(g, 1), 1, size(g, 2), size(g, 3)) for g in x.c])
    ket = TCI.TensorTrain([reshape(g, size(g, 1), size(g, 2), 1, size(g, 3)) for g in y.c])
    indices = fill((1, 1), length(x))
    if length(x) == 1
        # TCI 0.9.19's Contraction.evaluate assumes at least two sites.
        result = TCI.contract(bra, ket; algorithm = :zipup, tolerance = 0.0)
        return TCI.evaluate(result, indices)
    end
    TCI.evaluate(TCI.Contraction(bra, ket), indices)
end
ttnorm(x::QTT) = sqrt(max(0, real(inner(x, x))))
"Contract <x|W|y> without truncating or constructing W*y."
function inner_mpo(x::QTT, w, y::QTT)
    length(x) == length(y) == length(w) || throw(DimensionMismatch("MPO expectation lengths"))
    e = ones(ComplexF64, 1, 1, 1)
    for (a, m, b) in zip(x.c, w, y.c)
        @tensor t1[i, j, t, L] := e[i, j, l] * b[l, t, L]
        @tensor t2[i, s, J, L] := t1[i, j, t, L] * m[j, s, t, J]
        @tensor f[I, J, L] := conj(a[i, s, I]) * t2[i, s, J, L]
        e = f
    end
    e[1]
end
normalize(x::QTT) = scale(x, 1 / ttnorm(x))
function algebra_options(tol, cap)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("Invalid QTT tolerance"))
    cap >= 1 || throw(ArgumentError("QTT rank cap must be positive"))
    nothing
end
function rounded(x::QTT; tol = 1e-11, cap = 64)
    algebra_options(tol, cap)
    tt = with_svd_context(;
        operation = "rounded",
        original_cores = x.c,
        tolerance = tol,
        rank_cap = cap,
    ) do
        check_compression_shapes(x.c)
        result = deepcopy(tensortrain(x))
        guarded_compress!(result; tolerance = Float64(tol), maxbonddim = cap)
        result
    end
    QTT(tt.sitetensors)
end
"Pointwise addition uses recorded TCI when rec is supplied, otherwise library TT addition."
function add(x::QTT, y::QTT; a = 1, b = 1, tol = 1e-11, cap = 64, rec = nothing, label = "add")
    algebra_options(tol, cap)
    length(x) == length(y) || throw(DimensionMismatch("QTT addition lengths"))
    if rec !== nothing
        return linear_combination(rec, (x, y), (a, b); tol, cap, label)
    end
    # TCI 0.9.19's block addition assumes distinct first and last sites.
    length(x) == 1 && return QTT([a .* only(x.c) .+ b .* only(y.c)])
    # Same block construction and scaling site as TCI.add; route its
    # compression through the checked implementation of the same sweeps.
    d = length(x)
    result = TCI.TensorTrain([
        TCI._addtttensor(
            x.c[k],
            y.c[k];
            factorA = k == d ? a : one(ComplexF64),
            factorB = k == d ? b : one(ComplexF64),
            lefttensor = k == 1,
            righttensor = k == d,
        ) for k = 1:d
    ])
    original = copy(result.sitetensors)
    with_svd_context(;
        operation = "direct_add",
        original_cores = original,
        tolerance = tol,
        rank_cap = cap,
    ) do
        guarded_compress!(result; tolerance = Float64(tol), maxbonddim = cap)
    end
    QTT(result.sitetensors)
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
function gram(xs)
    g = zeros(ComplexF64, length(xs), length(xs))
    for j in eachindex(xs), i in firstindex(xs):j
        value = inner(xs[i], xs[j])
        g[i, j] = value
        i == j || (g[j, i] = conj(value))
    end
    g
end
function gram(xs, ys)
    g = zeros(ComplexF64, length(xs), length(ys))
    for j in eachindex(ys), i in eachindex(xs)
        g[i, j] = inner(xs[i], ys[j])
    end
    g
end
