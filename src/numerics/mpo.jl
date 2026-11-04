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
            for a = 1:rl, b = 1:rr
                w[a + rl * bin, i + 1, j + 1, b + rr * (k == d ? 0 : bout)] += g[a, s, b]
            end
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
function svd_tail_rank(s, cutoff, cap)
    r = length(s)
    discarded = 0.0
    while r > 1 && discarded + s[r]^2 <= cutoff^2
        discarded += s[r]^2
        r -= 1
    end
    min(r, cap)
end
"Apply an MPO; a proven right-canonical input can skip the consumer QR sweep."
function apply_mpo(
    w,
    x::QTT;
    tol = 1e-11,
    cap = 64,
    algorithm = :zipup,
    input_right_canonical = false,
)
    algebra_options(tol, cap)
    length(w) == length(x) || throw(DimensionMismatch("MPO action lengths"))
    result = if algorithm === :zipup
        # Local truncation must not depend on an arbitrary input-core gauge.
        # Exact zip-up needs no suffix metric; truncated zip-up uses a QR gauge.
        temporary_cap = max(2cap, 16)
        truncates =
            tol > 0 || any(k -> size(w[k], 4) * size(x.c[k], 3) > temporary_cap, 1:(length(x) - 1))
        input = truncates && !input_right_canonical ? right_canonical(x) : x
        cores = Array{ComplexF64,3}[]
        carry = ones(ComplexF64, 1, 1, 1)
        for (k, (a, b)) in enumerate(zip(w, input.c))
            @tensor g[v, s, U, V] := carry[v, u, l] * a[u, s, t, U] * b[l, t, V]
            if k == length(x)
                push!(cores, reshape(g, size(g, 1), 2, 1))
            else
                f = with_svd_context(;
                    operation = "mpo_zipup",
                    site = k,
                    direction = "forward",
                    original_cores = x.c,
                    operator = w,
                    tolerance = tol,
                    rank_cap = cap,
                ) do
                    stable_svd(reshape(g, size(g, 1) * 2, :))
                end
                r = svd_tail_rank(f.S, tol * norm(f.S) / length(x), max(2cap, 16))
                push!(cores, reshape(f.U[:, 1:r], size(g, 1), 2, r))
                carry = reshape(Diagonal(f.S[1:r]) * f.Vt[1:r, :], r, size(a, 4), size(b, 3))
            end
        end
        QTT(cores)
    else
        product = TCI.contract(
            TCI.TensorTrain(w),
            tensortrain(x);
            algorithm,
            tolerance = Float64(tol),
            maxbonddim = max(2cap, 16),
        )
        QTT(product.sitetensors)
    end
    rounded(result; tol, cap)
end
function delta_tt(R, j)
    QTT([reshape(ComplexF64[s == ((j >> (k - 1)) & 1) + 1 for s = 1:2], 1, 2, 1) for k = 1:R])
end
function green_mpo(R, h, mu)
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
    circulant(scale(kern, h^2 / (sinh(gamma) * (-expm1(-gamma * N)))))
end
