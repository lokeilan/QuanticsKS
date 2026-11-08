# Three Cartesian axes, interleaved from least to most significant bits.

"Cell-centered box [-Lx/2,Lx/2] x ... with zero exterior orbital amplitudes."
struct BoxGrid3D <: AbstractGrid
    R::Int
    L::NTuple{3,Float64}
    function BoxGrid3D(R::Integer, lengths)
        # The signed convolution grid needs three additional binary sites.
        2 <= R <= 19 || throw(ArgumentError("3D R must be 2:19 bits per axis"))
        ls = lengths isa Real ? ntuple(_ -> Float64(lengths), 3) : Tuple(Float64.(lengths))
        length(ls) == 3 && all(x -> isfinite(x) && x > 0, ls) ||
            throw(ArgumentError("Provide three positive box lengths in bohr"))
        new(Int(R), ls)
    end
end
Base.length(g::BoxGrid3D) = 1 << nsites(g)
nsites(g::BoxGrid3D) = 3g.R
spacing(g::BoxGrid3D) = g.L ./ (1 << g.R)
cell_volume(g::BoxGrid3D) = prod(g.L)
volume_element(g::BoxGrid3D) = prod(spacing(g))
Base.:(==)(a::BoxGrid3D, b::BoxGrid3D) = a.R == b.R && a.L == b.L

function grid_indices(g::BoxGrid3D, j::Integer)
    0 <= j < length(g) || throw(BoundsError(g, j))
    ntuple(a -> sum(((j >> (3b + a - 1)) & 1) << b for b = 0:(g.R - 1)), 3)
end
function grid_index(g::BoxGrid3D, indices)
    length(indices) == 3 && all(i -> i isa Integer && 0 <= i < 1 << g.R, indices) ||
        throw(ArgumentError("3D indices must be zero-based and inside the box"))
    sum(((indices[a] >> b) & 1) << (3b + a - 1) for b = 0:(g.R - 1), a = 1:3)
end
function grid_position(g::BoxGrid3D, j::Integer)
    indices, h = grid_indices(g, j), spacing(g)
    ntuple(a -> -g.L[a] / 2 + (indices[a] + 0.5) * h[a], 3)
end

abstract type AbstractModel3D <: AbstractModel end
"Isolated, restricted all-electron KS with unpolarized PZ81 LDA or Hartree only."
struct AllElectronLDA3D <: AbstractModel3D
    positions::Vector{NTuple{3,Float64}}
    charges::Vector{Float64}
    xc::Symbol
    function AllElectronLDA3D(positions, charges = ones(length(positions)); xc = :pz81)
        ps = [Tuple(Float64.(r)) for r in positions]
        zs = Float64.(charges)
        !isempty(ps) &&
            length(ps) == length(zs) &&
            all(r -> length(r) == 3 && all(isfinite, r), ps) &&
            all(z -> isfinite(z) && z > 0, zs) || throw(ArgumentError("Invalid 3D nuclei"))
        length(unique(ps)) == length(ps) || throw(ArgumentError("Coincident nuclei"))
        xc in (:pz81, :none) || throw(ArgumentError("3D XC must be :pz81 or :none"))
        new(ps, zs, xc)
    end
end
struct LocalPotential3D{F} <: AbstractModel3D
    potential::F
    id::String
    function LocalPotential3D(f; id::AbstractString)
        isempty(id) && throw(ArgumentError("A stable potential id is required"))
        new{typeof(f)}(f, String(id))
    end
end
function validate_model_grid(g::BoxGrid3D, m::AbstractModel3D, w, mass)
    if m isa AllElectronLDA3D
        all(==(2.0), w) || throw(ArgumentError("Restricted 3D KS requires double occupations"))
        m.xc == :pz81 && mass != 1 && throw(ArgumentError("PZ81 assumes electron mass 1"))
        all(r -> all(a -> abs(r[a]) < g.L[a] / 2, 1:3), m.positions) ||
            throw(ArgumentError("Nuclei must lie inside the box"))
    end
    nothing
end

const ANGSTROM_TO_BOHR = 1.8897261246257702
"Simple-cubic finite H cluster; paper spacing is 2 angstrom (Section 8.5)."
function hydrogen_cluster(;
    R = 7,
    shape = (3, 2, 2),
    distance = 2ANGSTROM_TO_BOHR,
    padding = 10.0,
    cell = nothing,
    xc = :pz81,
)
    length(shape) == 3 && all(n -> n isa Integer && n > 0, shape) && iseven(prod(shape)) ||
        throw(ArgumentError("Use three positive lattice sizes with an even atom count"))
    isfinite(distance) && distance > 0 && isfinite(padding) && padding > 0 ||
        throw(ArgumentError("Invalid cluster distance or padding"))
    positions = [
        ntuple(a -> (i[a] - (shape[a] - 1) / 2) * distance, 3) for
        i in Iterators.product((0:(n - 1) for n in shape)...)
    ]
    lengths = cell === nothing ? ntuple(a -> (shape[a] - 1) * distance + 2padding, 3) : cell
    Problem(BoxGrid3D(R, lengths), AllElectronLDA3D(vec(positions); xc), fill(2.0, prod(shape) ÷ 2))
end
model_descriptor(m::AllElectronLDA3D) = Dict(
    "kind" => "all_electron_lda_3d",
    "positions" => collect.(m.positions),
    "charges" => m.charges,
    "xc" => string(m.xc),
)
model_descriptor(m::LocalPotential3D) = Dict("kind" => "local_potential_3d", "id" => m.id)
function problem_descriptor(p::Problem{<:AbstractModel3D})
    Dict(
        "R" => p.grid.R,
        "cell" => collect(p.grid.L),
        "dimensions" => 3,
        "ordering" => "interleaved_lsb",
        "nodes" => "cell_centers",
        "occupations" => p.occupations,
        "mass" => p.mass,
        "model" => model_descriptor(p.model),
        "boundary" => "isolated_zero_exterior",
        "units" => "atomic",
    )
end
