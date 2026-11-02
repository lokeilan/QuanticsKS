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
function svd_tail_rank(s, cutoff, cap)
    r = length(s)
    discarded = 0.0
    while r > 1 && discarded + s[r]^2 <= cutoff^2
        discarded += s[r]^2
        r -= 1
    end
    min(r, cap)
end
function apply_mpo(w, x::QTT; tol = 1e-11, cap = 64, algorithm = :zipup)
    algebra_options(tol, cap)
    length(w) == length(x) || throw(DimensionMismatch("MPO action lengths"))
    result = if algorithm === :zipup
        # Restore the historical left-orthogonal SVD zip-up and its tail
        # cutoff. LU pivot truncation produced large Green-action errors on
        # rank-limited R16 states. stable_svd retries LAPACK failures with QR.
        cores = Array{ComplexF64,3}[]
        carry = ones(ComplexF64, 1, 1, 1)
        for (k, (a, b)) in enumerate(zip(w, x.c))
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
    gamma = 2asinh(h * sqrt(mu / 2))
    N = 2^R
    # Two rank-one exponentials; never form a vector of length N.
    p = QTT([reshape(ComplexF64[1, exp(-gamma * 2^(k - 1))], 1, 2, 1) for k = 1:R])
    q = QTT([reshape(ComplexF64[exp(-gamma * 2^(k - 1)), 1], 1, 2, 1) for k = 1:R])
    kern = add(p, scale(q, exp(-gamma)); tol = 1e-14, cap = 2)
    circulant(scale(kern, h^2 / (sinh(gamma) * (-expm1(-gamma * N)))))
end
