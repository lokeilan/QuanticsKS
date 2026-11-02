# FCC indices, block storage, and expansion; saved phases recover the original samples.

# FCC maps and boundary convention follow silicon Step1's joint_expand/restrict.
# Dynamics, cold initialization, and SCF use the QuanticsKS backend in this project.
import TensorCrossInterpolation as TCI
const FCC_CHARACTERS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
export FCCState

fcc_bits(l) = l.space_bits .- (1, 1, 0)
fcc_shape(l) = shape(l) .÷ (2, 2, 1)
fcc_layout(l) = JointLayout(fcc_bits(l), l.k_bits; a_bohr = l.a_bohr, k_shift = l.k_shift)
fcc_sites(l) = sum(fcc_bits(l)) + sum(l.k_bits) + 2
fcc_klabel(l, k) = ntuple(a -> mod(k[a] + kshape(l)[a] ÷ 2, kshape(l)[a]), 3)
fcc_kindex(l, k) = LinearIndices(kshape(l))[(fcc_klabel(l, k) .+ 1)...]

function fcc_expand(v, chi)
    nx, ny, nz = size(v)
    full = zeros(ComplexF64, 2nx, 2ny, nz)
    for sx = 0:1, sy = 0:1, i = 1:nx, j = 1:ny, k = 1:nz
        full[i + sx * nx, j + sy * ny, mod1(k + (sx + sy) * (nz ÷ 2), nz)] =
            0.5 * chi[1]^sx * chi[2]^sy * v[i, j, k]
    end
    full
end
function fcc_restrict(full, chi)
    nx, ny, nz = size(full)
    nx ÷= 2
    ny ÷= 2
    v = zeros(ComplexF64, nx, ny, nz)
    for sx = 0:1, sy = 0:1, i = 1:nx, j = 1:ny, k = 1:nz
        v[i, j, k] +=
            0.5 *
            chi[1]^sx *
            chi[2]^sy *
            full[i + sx * nx, j + sy * ny, mod1(k + (sx + sy) * (nz ÷ 2), nz)]
    end
    v
end
fcc_expand_matrix(l, V, chi) =
    hcat([vec(fcc_expand(reshape(view(V, :, b), fcc_shape(l)), chi)) for b in axes(V, 2)]...)
fcc_restrict_matrix(l, U, chi) =
    hcat([vec(fcc_restrict(reshape(view(U, :, b), shape(l)), chi)) for b in axes(U, 2)]...)

function fcc_encode(l, x, k, b)
    1 <= b <= 4 || error("FCC band outside 1:4")
    d = sum(fcc_bits(l)) + sum(l.k_bits)
    z = b - 1
    encode(fcc_layout(l), x, fcc_klabel(l, k)) | ((z >> 1) << d) | ((z & 1) << (d + 1))
end
function fcc_decode(l, label)
    0 <= label < 1 << fcc_sites(l) || error("FCC label outside joint grid")
    d = sum(fcc_bits(l)) + sum(l.k_bits)
    x, k = decode(fcc_layout(l), label & ((1 << d) - 1))
    x, fcc_klabel(l, k), 1 + 2 * ((label >> d) & 1) + ((label >> (d + 1)) & 1)
end

struct FCCState
    layout::JointLayout
    blocks::Vector{Q.QTT}
    phases::Array{ComplexF64,3} # k, band, block; undo for raw block identities.
    function FCCState(l, blocks, phases)
        length(blocks) == 4 && size(phases) == (prod(kshape(l)), 4, 4) ||
            error("FCC state dimensions")
        all(x -> length(x) == fcc_sites(l), blocks) || error("FCC joint site order")
        all(x -> all(g -> size(g, 2) == 2 && all(isfinite, g), x.c), blocks) ||
            error("FCC binary cores")
        all(x -> size(first(x.c), 1) == size(last(x.c), 3) == 1, blocks) ||
            error("FCC endpoint ranks")
        all(x -> all(i -> size(x.c[i], 3) == size(x.c[i + 1], 1), 1:(length(x) - 1)), blocks) ||
            error("FCC bond continuity")
        maximum(abs.(abs.(phases) .- 1)) < 1e-10 || error("FCC phases must be unit modulus")
        new(l, collect(blocks), phases)
    end
end

function spatial_factors(s::FCCState)
    l = fcc_layout(s.layout)
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
        end for x in s.blocks
    ]
end
function fcc_slice(s::FCCState, factors, k, sector; raw = true)
    l = s.layout
    train = s.blocks[sector]
    start = sum(fcc_bits(l))
    values = hcat(
        [
            begin
                label = fcc_encode(l, (0, 0, 0), k, b)
                tail = ones(ComplexF64, 1)
                for site = length(train):-1:(start + 1)
                    tail = train.c[site][:, ((label >> (site - 1)) & 1) + 1, :] * tail
                end
                factors[sector] * tail
            end for b = 1:4
        ]...,
    )
    raw || return values
    ik = fcc_kindex(l, k)
    values ./ transpose(s.phases[ik, :, sector])
end
function state_slice(s::FCCState, factors, k)
    hcat(
        [
            fcc_expand_matrix(s.layout, fcc_slice(s, factors, k, b), FCC_CHARACTERS[b]) for b = 1:4
        ]...,
    )
end
function storage_report(s::FCCState)
    profiles = [[size(g, 3) for g in x.c[1:(end - 1)]] for x in s.blocks]
    bytes = sum(Q.storage, s.blocks)
    (;
        representation = "fcc_joint_xk_band",
        orbital_count = 16,
        train_count = 4,
        joint_sites = fcc_sites(s.layout),
        spatial_sites = sum(fcc_bits(s.layout)),
        k_sites = sum(s.layout.k_bits),
        band_sites = 2,
        core_bytes = bytes,
        phase_bytes = sizeof(s.phases),
        retained_bytes = bytes + sizeof(s.phases),
        maximum_bond = maximum(Q.rank, s.blocks),
        bond_profiles = profiles,
        dense_orbital_bytes = 16 * 16prod(shape(s.layout)) * prod(kshape(s.layout)),
        dense_quotient_bytes = 16 * 16prod(fcc_shape(s.layout)) * prod(kshape(s.layout)),
    )
end
state_trains(s::JointState) = s.orbitals
state_trains(s::FCCState) = s.blocks
truncate_state(s::JointState, cap) =
    JointState(s.layout, [Q.rounded(x; tol = 0.0, cap) for x in s.orbitals])
truncate_state(s::FCCState, cap) =
    FCCState(s.layout, [Q.rounded(x; tol = 0.0, cap) for x in s.blocks], s.phases)

include("fcc_sampling.jl")
