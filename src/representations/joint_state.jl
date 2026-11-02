# Store physical orbitals u(x,k); slices, Gram matrices, and densities use volume weight dV.

"Physical u_b(x,k) samples, one joint QTT per band; no implicit norm repair."
struct JointState
    layout::JointLayout
    orbitals::Vector{Q.QTT}
    function JointState(l::JointLayout, orbitals)
        !isempty(orbitals) || throw(ArgumentError("empty occupied state"))
        for x in orbitals
            length(x) == sites(l) || throw(DimensionMismatch("joint core count"))
            size(first(x.c), 1) == size(last(x.c), 3) == 1 ||
                throw(DimensionMismatch("TT endpoint ranks"))
            all(g -> size(g, 2) == 2 && all(isfinite, g), x.c) ||
                throw(ArgumentError("finite binary TT cores required"))
            all(i -> size(x.c[i], 3) == size(x.c[i + 1], 1), 1:(length(x) - 1)) ||
                throw(DimensionMismatch("TT bond ranks"))
        end
        new(l, collect(orbitals))
    end
end

# Contract fixed k bits into the open spatial right bond. No spatial vector is built.
function ktail(l, x, k)
    label = encode(l, (0, 0, 0), k)
    env = ones(ComplexF64, 1)
    for site = sites(l):-1:(sum(l.space_bits) + 1)
        bit = ((label >> (site - 1)) & 1) + 1
        env = x.c[site][:, bit, :] * env
    end
    env
end

"Physical spatial Gram matrix at one k; k weights do not enter this constraint."
function gram_at(s::JointState, k)
    l = s.layout
    n = length(s.orbitals)
    S = zeros(ComplexF64, n, n)
    tails = [ktail(l, x, k) for x in s.orbitals]
    for b = 1:n, c = b:n
        env = ones(ComplexF64, 1, 1)
        for site = 1:sum(l.space_bits)
            g = s.orbitals[b].c[site]
            h = s.orbitals[c].c[site]
            env = sum(g[:, v, :]' * env * h[:, v, :] for v = 1:2)
        end
        S[b, c] = dv(l) * dot(tails[b], env * tails[c])
        S[c, b] = conj(S[b, c])
    end
    S
end

"Uniform normalized k mesh; exact k-tail contraction at a spatial sample."
function density_at(s::JointState, x; occupations = fill(2.0, length(s.orbitals)))
    length(occupations) == length(s.orbitals) &&
        all(f -> isfinite(f) && f >= 0, occupations) &&
        any(>(0), occupations) ||
        throw(ArgumentError("finite nonnegative occupations, with a nonzero total, required"))
    l = s.layout
    label = encode(l, x, (0, 0, 0))
    rho = 0.0
    for (orbital, f) in zip(s.orbitals, occupations)
        left = ones(ComplexF64, 1)
        for site = 1:sum(l.space_bits)
            bit = ((label >> (site - 1)) & 1) + 1
            left = transpose(orbital.c[site][:, bit, :]) * left
        end
        env = conj.(left) * transpose(left)
        for site = (sum(l.space_bits) + 1):sites(l)
            g = orbital.c[site]
            env = sum(g[:, v, :]' * env * g[:, v, :] for v = 1:2) / 2
        end
        rho += f * real(only(env))
    end
    rho
end

"Reconstruct sum_b u_b(x,k) C_bc(k) by a joint TCI, without global normalization."
function rotate_joint(
    s::JointState,
    coefficients;
    tolerance = 1e-11,
    cap = 64,
    tci = TCIOptions(),
    recorder = nothing,
)
    n = length(s.orbitals)
    evals = Q.Evaluator.(s.orbitals)
    functions = [
        (x, k) -> begin
            C = coefficients(k)
            size(C) == (n, n) || throw(DimensionMismatch("square per-k rotation required"))
            sum(evals[b](encode(s.layout, x, k)) * C[b, c] for b = 1:n)
        end for c = 1:n
    ]
    sample_joint(s.layout, functions; tolerance, cap, tci, recorder)
end

function storage_report(s::JointState)
    profiles = [[size(g, 3) for g in x.c[1:(end - 1)]] for x in s.orbitals]
    (;
        orbital_count = length(s.orbitals),
        joint_sites = sites(s.layout),
        core_bytes = sum(Q.storage, s.orbitals),
        maximum_bond = maximum(Q.rank, s.orbitals),
        bond_profiles = profiles,
        dense_orbital_bytes = 16 *
                              length(s.orbitals) *
                              prod(shape(s.layout)) *
                              prod(kshape(s.layout)),
    )
end
"Exact spatial prefix matrices, used only as bounded contraction workspaces."
function spatial_factors(s::JointState)
    l = s.layout
    rows =
        [encode(l, Tuple(I) .- 1, (0, 0, 0)) + 1 for I in vec(collect(CartesianIndices(shape(l))))]
    [
        begin
            A = ones(ComplexF64, 1, 1)
            for site = 1:sum(l.space_bits)
                g = x.c[site]
                A = vcat(A * g[:, 1, :], A * g[:, 2, :])
            end
            A[rows, :]
        end for x in s.orbitals
    ]
end
function state_slice(s, factors, k)
    hcat([factors[b] * ktail(s.layout, x, k) for (b, x) in enumerate(s.orbitals)]...)
end
kcoordinates(l) = vec([Tuple(I) .- 1 for I in CartesianIndices(kshape(l))])

include("joint_sampling.jl")
