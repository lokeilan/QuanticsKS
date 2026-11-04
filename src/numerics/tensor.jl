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
const DIRECT_INNER_SITE_WORK_LIMIT = 4 * 48^3
const DIRECT_INNER_PARALLEL_SITE_WORK = 4 * 40^3
const DIRECT_INNER_PARALLEL_WORK = 4 * DIRECT_INNER_PARALLEL_SITE_WORK

inner_site_work(gx, gy) = size(gx, 2) * size(gy, 1) * size(gx, 3) * (size(gx, 1) + size(gy, 3))

"Select from both bond profiles; keep large contractions on the TCI/BLAS path."
function inner_backend(x::QTT, y::QTT)
    length(x) == length(y) || throw(DimensionMismatch("QTT inner-product lengths"))
    isempty(x.c) && throw(ArgumentError("An inner product needs nonempty tensor trains"))
    left_x = left_y = 1
    direct = true
    for (gx, gy) in zip(x.c, y.c)
        rx, p, ox = size(gx)
        ry, q, oy = size(gy)
        rx == left_x && ry == left_y && p == q ||
            throw(DimensionMismatch("QTT inner-product bonds or physical dimensions"))
        direct &= inner_site_work(gx, gy) <= DIRECT_INNER_SITE_WORK_LIMIT
        left_x, left_y = ox, oy
    end
    left_x == left_y == 1 || throw(DimensionMismatch("QTT inner-product boundaries"))
    direct ? :direct : :tci
end

function inner_direct_serial(x::QTT, y::QTT)
    buffer = ComplexF64[1]
    work = ComplexF64[]
    left_x = 1
    for (gx, gy) in zip(x.c, y.c)
        rx, p, ox = size(gx)
        ry, _, oy = size(gy)
        resize!(work, ry * p * ox)
        @inbounds for s = 1:p, m = 1:ry, a = 1:ox
            value = zero(ComplexF64)
            for l = 1:rx
                value += buffer[l + (m - 1) * left_x] * conj(gx[l, s, a])
            end
            work[(s - 1) * ry + m + (a - 1) * p * ry] = value
        end
        resize!(buffer, ox * oy)
        fill!(buffer, 0)
        @inbounds for s = 1:p, m = 1:ry, a = 1:ox
            value = work[(s - 1) * ry + m + (a - 1) * p * ry]
            iszero(value) && continue
            for b = 1:oy
                buffer[a + (b - 1) * ox] += value * gy[m, s, b]
            end
        end
        left_x = ox
    end
    only(buffer)
end

function inner_site_workers(gx, gy, cfg)
    inner_site_work(gx, gy) < DIRECT_INNER_PARALLEL_SITE_WORK && return 1
    parallel_workers(cfg, size(gx, 3))
end

function inner_direct_workers(x, y, cfg)
    parallel_workers(cfg, 2) <= 1 && return 1
    work = 0
    workers = 1
    for (gx, gy) in zip(x.c, y.c)
        count = inner_site_workers(gx, gy, cfg)
        if count > 1
            work += inner_site_work(gx, gy)
            workers = max(workers, count)
        end
    end
    work >= DIRECT_INNER_PARALLEL_WORK ? workers : 1
end

"Write independent output rows; each worker owns its intermediate column."
function inner_direct_site!(output, input, gx, gy, scratch, cfg)
    rx, p, ox = size(gx)
    ry, _, oy = size(gy)
    workers = inner_site_workers(gx, gy, cfg)
    foreach_independent(workers, (; parallel_tasks = workers)) do worker
        work = scratch[worker]
        first_a = fld((worker - 1) * ox, workers) + 1
        last_a = fld(worker * ox, workers)
        @inbounds for a = first_a:last_a
            for s = 1:p, m = 1:ry
                value = zero(ComplexF64)
                for l = 1:rx
                    value += input[l + (m - 1) * rx] * conj(gx[l, s, a])
                end
                work[(s - 1) * ry + m] = value
            end
            for s = 1:p, m = 1:ry
                value = work[(s - 1) * ry + m]
                iszero(value) && continue
                for b = 1:oy
                    output[a + (b - 1) * ox] += value * gy[m, s, b]
                end
            end
        end
    end
    output
end

function inner_direct(x::QTT, y::QTT; cfg = (;))
    workers = inner_direct_workers(x, y, cfg)
    workers <= 1 && return inner_direct_serial(x, y)
    width = maximum(size(gx, 2) * size(gy, 1) for (gx, gy) in zip(x.c, y.c))
    scratch = [Vector{ComplexF64}(undef, width) for _ = 1:workers]
    input = ComplexF64[1]
    output = ComplexF64[]
    for (gx, gy) in zip(x.c, y.c)
        resize!(output, size(gx, 3) * size(gy, 3))
        fill!(output, 0)
        inner_direct_site!(output, input, gx, gy, scratch, cfg)
        input, output = output, input
    end
    only(input)
end

function inner_tci(x::QTT, y::QTT)
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
inner(x::QTT, y::QTT; cfg = (;)) =
    inner_backend(x, y) == :direct ? inner_direct(x, y; cfg) : inner_tci(x, y)
ttnorm(x::QTT; cfg = (;)) = sqrt(max(0, real(inner(x, x; cfg))))
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
normalize(x::QTT; cfg = (;)) = scale(x, 1 / ttnorm(x; cfg))
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

"Establish a right-orthogonal suffix by QR, without tolerance-based truncation."
function right_canonical(x::QTT)
    check_compression_shapes(x.c)
    cores = deepcopy(x.c)
    for (k, g) in enumerate(cores)
        with_svd_context(; operation = "right_canonical_qr", site = k) do
            require_finite(g, "nonfinite_right_canonical_input")
        end
    end
    for k = length(x):-1:2
        with_svd_context(; operation = "right_canonical_qr", site = k, original_cores = x.c) do
            g = cores[k]
            A = reshape(g, size(g, 1), :)
            factor = try
                qr(Matrix(A'))
            catch error
                event = save_svd_event(
                    A,
                    "right_canonical_qr_exception";
                    details = (; exception = sprint(showerror, error)),
                )
                finish_svd_event(event, "failed")
                rethrow()
            end
            triangular = Matrix(factor.R)
            r = size(triangular, 1)
            basis = factor.Q * Matrix{ComplexF64}(I, size(A, 2), r)
            require_finite(basis, "nonfinite_right_canonical_basis")
            require_finite(triangular, "nonfinite_right_canonical_factor")
            cores[k] = reshape(Matrix(basis'), r, size(g, 2), size(g, 3))
            previous = cores[k - 1]
            contracted = reshape(previous, :, size(previous, 3)) * triangular'
            require_finite(contracted, "nonfinite_right_canonical_contraction")
            cores[k - 1] = reshape(contracted, size(previous, 1), size(previous, 2), r)
        end
    end
    QTT(cores)
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
function gram(xs; cfg = (;))
    g = zeros(ComplexF64, length(xs), length(xs))
    foreach_independent(length(xs), cfg) do j
        for i = firstindex(xs):j
            value = inner(xs[i], xs[j]; cfg)
            g[i, j] = value
            i == j || (g[j, i] = conj(value))
        end
    end
    g
end
function gram(xs, ys; cfg = (;))
    g = zeros(ComplexF64, length(xs), length(ys))
    foreach_independent(length(ys), cfg) do j
        for i in eachindex(xs)
            g[i, j] = inner(xs[i], ys[j]; cfg)
        end
    end
    g
end
