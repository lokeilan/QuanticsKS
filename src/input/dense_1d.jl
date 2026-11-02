# Dense 1D options: lengths in bohr, energies and eigenpair residuals in Ha.

Base.@kwdef struct Config
    R::Int = 16
    natoms::Int = 100
    spacing::Float64 = 4.0
    extra_states::Int = 8
    mixing::Float64 = 0.15
    max_scf::Int = 200
    max_eigen::Int = 500
    eigen_tol::Float64 = 1e-8
    density_tol::Float64 = 1e-8
    residual_tol::Float64 = 1e-6
    energy_tol::Float64 = 1e-8
    gram_tol::Float64 = 1e-8
    charge_tol::Float64 = 1e-8
    degeneracy_tol::Float64 = 1e-6
    preconditioner_shift::Float64 = 0.5
    xc_rtol::Float64 = 1e-10
    seed::Int = 20261030
    blas_threads::Int = 1
    fft_threads::Int = 1
    julia_threads::Int = 1
    mixing_method::String = "linear"
    pulay_history::Int = 6
    eigensolver::String = "cyclic"
    fermi_pair_tolerance::Float64 = 1e-10
    arpack_tol::Float64 = 1e-12
    spectral_shift_margin::Float64 = 0.05
    # Explicit legacy default keeps old saved inputs reproducible. Production
    # h100-r*.toml selects the standard QTT model.
    physical_model::String = "minimum_image"
    xc_table_points::Int = 2048
    xc_density_min::Float64 = 1e-8
    xc_density_max::Float64 = 4.0
    exchange_quadrature_rtol::Float64 = 2e-11
end

function validate(c::Config)
    4 <= c.R <= 20 || error("R must be in 4:20")
    c.natoms >= 2 && iseven(c.natoms) || error("natoms must be positive and even")
    c.extra_states >= 2 || error("At least two unoccupied states are required")
    3 * (c.natoms ÷ 2 + c.extra_states) < 2^c.R || error("Grid too small for the LOBPCG block")
    all(isfinite, (getfield(c, k) for k in fieldnames(Config) if getfield(c, k) isa Number)) ||
        error("Nonfinite input")
    c.spacing > 0 && 0 < c.mixing <= 1 || error("Invalid spacing or mixing")
    c.mixing_method in ("linear", "pulay") || error("Unknown density mixing method")
    c.eigensolver in ("block", "cyclic") || error("Unknown eigensolver")
    c.physical_model in ("minimum_image", "qtt_periodic") || error("Unknown physical model")
    c.xc_table_points >= 4 &&
        0 < c.xc_density_min < c.xc_density_max &&
        0 < c.exchange_quadrature_rtol < 1 || error("Invalid QTT XC numerics")
    c.pulay_history >= 2 || error("Pulay history must be at least two")
    all(>(0), (c.max_scf, c.max_eigen, c.blas_threads, c.fft_threads, c.julia_threads)) ||
        error("Invalid iteration/thread count")
    all(
        >(0),
        (
            c.eigen_tol,
            c.density_tol,
            c.residual_tol,
            c.energy_tol,
            c.gram_tol,
            c.charge_tol,
            c.degeneracy_tol,
            c.preconditioner_shift,
            c.xc_rtol,
        ),
    ) || error("Tolerances and shift must be positive")
    return c
end
load_config(path) = validate(Config(; (Symbol(k) => v for (k, v) in TOML.parsefile(path))...))
config_dict(c) = Dict(string(k) => getfield(c, k) for k in fieldnames(Config))
