# Map zero-based coordinates to QTT labels, with most-significant bits first per axis.

"Space then k, axis grouped, MSB first within each axis; zero-based coordinates."
struct JointLayout
    space_bits::NTuple{3,Int}
    k_bits::NTuple{3,Int}
    a_bohr::Float64
    k_shift::NTuple{3,Float64}
    function JointLayout(space_bits, k_bits; a_bohr = 10.26, k_shift = (0.5, 0.5, 0.5))
        length(space_bits) == length(k_bits) == length(k_shift) == 3 ||
            throw(ArgumentError("three spatial axes and three k axes required"))
        s = Tuple(Int.(space_bits))
        k = Tuple(Int.(k_bits))
        shift = Tuple(Float64.(k_shift))
        all(>(0), s) && all(>=(0), k) && sum(s) + sum(k) <= 50 ||
            throw(ArgumentError("positive spatial bits, nonnegative k bits, total <= 50 required"))
        isfinite(a_bohr) && a_bohr > 0 && all(x -> isfinite(x) && 0 <= x < 1, shift) ||
            throw(ArgumentError("invalid cell or mesh-step k shift"))
        new(s, k, Float64(a_bohr), shift)
    end
end

sites(l::JointLayout) = sum(l.space_bits) + sum(l.k_bits)
shape(l::JointLayout) = 2 .^ l.space_bits
kshape(l::JointLayout) = 2 .^ l.k_bits
dv(l::JointLayout) = l.a_bohr^3 / prod(shape(l))

"Pack MSB-ordered tensor sites into the vendor Evaluator's LSB-ordered label."
function encode(l::JointLayout, x, k)
    length(x) == length(k) == 3 || throw(ArgumentError("three coordinates required"))
    result = 0
    site = 0
    for (coordinate, bits) in zip((x..., k...), (l.space_bits..., l.k_bits...))
        coordinate isa Integer && 0 <= coordinate < (1 << bits) ||
            throw(ArgumentError("coordinate outside joint grid"))
        for b = (bits - 1):-1:0
            result |= ((coordinate >> b) & 1) << site
            site += 1
        end
    end
    result
end

function decode(l::JointLayout, label::Integer)
    0 <= label < (1 << sites(l)) || throw(ArgumentError("joint label out of range"))
    bitcounts = (l.space_bits..., l.k_bits...)
    coordinates = ntuple(6) do axis
        site = sum(bitcounts[j] for j = 1:(axis - 1); init = 0)
        value = 0
        for _ = 1:bitcounts[axis]
            value = (value << 1) | ((label >> site) & 1)
            site += 1
        end
        value
    end
    (
        (coordinates[1], coordinates[2], coordinates[3]),
        (coordinates[4], coordinates[5], coordinates[6]),
    )
end

function kpoint(l::JointLayout, k)
    encode(l, (0, 0, 0), k) # range check
    ntuple(a -> mod((k[a] + l.k_shift[a]) / kshape(l)[a] + 0.5, 1.0) - 0.5, 3)
end
