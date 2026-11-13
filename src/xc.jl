# Spin-resolved Libxc conversion, interaction parameters, and native-handle lifetime.

mutable struct SpinXC1D
    functions::Vector{Ptr{Cvoid}}
    library::Ptr{Cvoid}
    mode::Symbol
    spin::Symbol
    version::String
end

function SpinXC1D(model::SoftCoulomb1D)
    identifiers =
        model.xc == :libxc_1d_lda ? (:lda_x_1d_soft, :lda_c_1d_csc) :
        model.xc in (:libxc_1d_x, :helbig_1d_lsda) ? (:lda_x_1d_soft,) : ()
    isempty(identifiers) && return SpinXC1D(Ptr{Cvoid}[], C_NULL, model.xc, model.spin, "not used")
    path = get(ENV, "QUANTICSKS_LIBXC", Libdl.find_library(["libxc", "xc"]))
    isempty(path) &&
        throw(ArgumentError("Set QUANTICSKS_LIBXC to the required Libxc shared library"))
    library = Libdl.dlopen(path)
    version = unsafe_string(ccall(Libdl.dlsym(library, :xc_version_string), Cstring, ()))
    backend = SpinXC1D(Ptr{Cvoid}[], library, model.xc, model.spin, version)
    finalizer(backend) do value
        for pointer in value.functions
            ccall(Libdl.dlsym(value.library, :xc_func_end), Cvoid, (Ptr{Cvoid},), pointer)
            ccall(Libdl.dlsym(value.library, :xc_func_free), Cvoid, (Ptr{Cvoid},), pointer)
        end
        Libdl.dlclose(value.library)
    end
    for id in identifiers
        pointer = ccall(Libdl.dlsym(library, :xc_func_alloc), Ptr{Cvoid}, ())
        pointer == C_NULL && error("Libxc allocation failed")
        status = ccall(
            Libdl.dlsym(library, :xc_func_init),
            Cint,
            (Ptr{Cvoid}, Cint, Cint),
            pointer,
            id == :lda_x_1d_soft ? 21 : 18,
            2,
        )
        if status != 0
            ccall(Libdl.dlsym(library, :xc_func_free), Cvoid, (Ptr{Cvoid},), pointer)
            error("Libxc initialization failed for $id")
        end
        push!(backend.functions, pointer)
        parameters = id == :lda_x_1d_soft ? [1.0] : [1.0, 1.0]
        ccall(
            Libdl.dlsym(library, :xc_func_set_ext_params),
            Cvoid,
            (Ptr{Cvoid}, Ptr{Cdouble}),
            pointer,
            parameters,
        )
        ccall(
            Libdl.dlsym(library, :xc_func_set_dens_threshold),
            Cvoid,
            (Ptr{Cvoid}, Cdouble),
            pointer,
            1e-14,
        )
    end
    backend
end

function evaluate_spin_xc(backend::SpinXC1D, rho)
    n = length(rho)
    spin_density = zeros(2, n)
    if backend.spin == :unpolarized
        spin_density[1, :] .= rho ./ 2
        spin_density[2, :] .= rho ./ 2
    else
        spin_density[backend.spin == :up ? 1 : 2, :] .= rho
    end
    energy = zeros(n)
    potential = zeros(2, n)
    temporary_energy = zeros(n)
    temporary_potential = zeros(2, n)
    GC.@preserve backend begin
        for pointer in backend.functions
            ccall(
                Libdl.dlsym(backend.library, :xc_lda_exc_vxc),
                Cvoid,
                (Ptr{Cvoid}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                pointer,
                n,
                spin_density,
                temporary_energy,
                temporary_potential,
            )
            energy .+= temporary_energy .* rho
            potential .+= temporary_potential
        end
    end
    if backend.mode in (:helbig_1d_lsda, :restricted_exx_helbig_correlation)
        for index = 1:n
            correlation_energy, potential_up, potential_down =
                Q.helbig_spin_correlation(spin_density[1, index], spin_density[2, index])
            energy[index] += correlation_energy
            potential[1, index] += potential_up
            potential[2, index] += potential_down
        end
    end
    energy, vec(potential[backend.spin == :down ? 2 : 1, :])
end
