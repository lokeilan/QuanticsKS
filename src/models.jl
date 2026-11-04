# Periodic 1D grids, models, and occupations; Problem uses atomic units.

"Uniform periodic grid x_j=j*L/2^R, with atomic units throughout."
struct PeriodicGrid1D
    R::Int
    L::Float64
    function PeriodicGrid1D(R::Integer, L::Real)
        2 <= R <= 40 || throw(ArgumentError("R must be between 2 and 40"))
        isfinite(L) && L > 0 || throw(ArgumentError("Cell length must be finite and positive"))
        new(R, L)
    end
end
Base.length(g::PeriodicGrid1D) = 1 << g.R
spacing(g::PeriodicGrid1D) = g.L / length(g)
Base.:(==)(a::PeriodicGrid1D, b::PeriodicGrid1D) = a.R == b.R && a.L == b.L

abstract type AbstractModel1D end
"Periodic a=1 soft Coulomb and unpolarized 1D LSDA; fixed nuclei."
struct SoftCoulombLSDA1D <: AbstractModel1D
    positions::Vector{Float64}
    charges::Vector{Float64}
    softening::Float64
    coupling::Float64
    xc::Symbol
    function SoftCoulombLSDA1D(
        positions,
        charges = ones(length(positions));
        softening = 1.0,
        coupling = 1.0,
        xc = :lsda,
    )
        !isempty(positions) && length(positions) == length(charges) ||
            throw(ArgumentError("Nuclear arrays must have equal positive length"))
        all(isfinite, positions) && all(z -> isfinite(z) && z > 0, charges) ||
            throw(ArgumentError("Invalid nuclei"))
        isfinite(softening) && softening > 0 && isfinite(coupling) && coupling > 0 ||
            throw(ArgumentError("Invalid interaction parameters"))
        xc in (:lsda, :none) || throw(ArgumentError("XC must be :lsda or :none"))
        xc == :lsda &&
            (softening != 1 || coupling != 1) &&
            throw(
                ArgumentError(
                    "This LSDA parametrization is valid only for softening=coupling=1; use xc=:none for Hartree experiments",
                ),
            )
        new(collect(positions), collect(charges), softening, coupling, xc)
    end
end
"Noninteracting particles in a real periodic potential f(x); id identifies its physical definition."
struct LocalPotential1D{F} <: AbstractModel1D
    potential::F
    id::String
    function LocalPotential1D(f; id::AbstractString)
        isempty(id) && throw(ArgumentError("A stable potential id is required for checkpoints"))
        new{typeof(f)}(f, id)
    end
end
struct ConstantPotential
    value::Float64
end
(f::ConstantPotential)(x) = f.value
function LocalPotential1D(value::Real; id = "constant:$value")
    isfinite(value) || throw(ArgumentError("Constant potential must be finite"))
    LocalPotential1D(ConstantPotential(value); id)
end
struct Problem{M<:AbstractModel1D}
    grid::PeriodicGrid1D
    model::M
    occupations::Vector{Float64}
    mass::Float64
    function Problem(
        grid::PeriodicGrid1D,
        model::M,
        occupations;
        mass = 1.0,
    ) where {M<:AbstractModel1D}
        isfinite(mass) && mass > 0 || throw(ArgumentError("Mass must be positive"))
        w = Float64.(collect(occupations))
        0 < length(w) <= length(grid) && all(x -> isfinite(x) && x > 0, w) ||
            throw(ArgumentError("Invalid occupations"))
        if model isa SoftCoulombLSDA1D
            model.softening <= grid.L / 4 || throw(
                ArgumentError("The current 28-term periodic kernel requires softening <= cell/4"),
            )
            model.xc == :lsda &&
                mass != 1 &&
                throw(ArgumentError("This LSDA parametrization assumes electron mass 1"))
            all(==(2.0), w) || throw(
                ArgumentError(
                    "Restricted unpolarized LSDA currently requires occupation 2 for every orbital",
                ),
            )
            isapprox(sum(w), sum(model.charges); atol = 1e-12, rtol = 0) ||
                throw(ArgumentError("Periodic soft-Coulomb systems must be neutral"))
            positions = mod.(model.positions, grid.L)
            length(unique(positions)) == length(positions) ||
                throw(ArgumentError("Coincident nuclei"))
        end
        new{M}(grid, model, w, mass)
    end
end
function hydrogen_chain(;
    R = 16,
    natoms = 100,
    distance = 4.0,
    cell = natoms * distance,
    offset = 0.0,
    softening = 1.0,
    coupling = 1.0,
    xc = :lsda,
    mass = 1.0,
)
    natoms > 0 && iseven(natoms) || throw(ArgumentError("Use a positive even number of H atoms"))
    Problem(
        PeriodicGrid1D(R, cell),
        SoftCoulombLSDA1D(offset .+ distance .* collect(0:(natoms - 1)); softening, coupling, xc),
        fill(2.0, natoms ÷ 2);
        mass,
    )
end

model_descriptor(m::SoftCoulombLSDA1D) = Dict(
    "kind" => "soft_coulomb_lsda_1d",
    "positions" => m.positions,
    "charges" => m.charges,
    "softening" => m.softening,
    "coupling" => m.coupling,
    "xc" => string(m.xc),
)
model_descriptor(m::LocalPotential1D) = Dict("kind" => "local_potential_1d", "id" => m.id)
function problem_descriptor(p)
    Dict(
        "R" => p.grid.R,
        "cell" => p.grid.L,
        "occupations" => p.occupations,
        "mass" => p.mass,
        "model" => model_descriptor(p.model),
        "boundary" => "periodic",
        "units" => "atomic",
    )
end
function compatible_problem(a, b; refinement = false)
    a.grid.L == b.grid.L &&
        a.mass == b.mass &&
        (refinement || a.grid.R == b.grid.R) &&
        a.occupations == b.occupations &&
        model_descriptor(a.model) == model_descriptor(b.model)
end
