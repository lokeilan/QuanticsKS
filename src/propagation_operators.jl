# Shared discrete kinetic propagators; no full propagator matrix is constructed for QTT.

propagation_orbital_tolerance(accuracy) =
    something(accuracy.orbital_tolerance, accuracy.storage_tolerance)
kinetic_grid(grid::LineGrid1D) = grid
kinetic_grid(grid::PeriodicGrid1D) = LineGrid1D(
    length(grid),
    spacing(grid);
    origin = 0.0,
    boundary = :periodic,
    fd_order = 2,
    cell = grid.L,
)

split_step_coefficients(time::TimeOptions) =
    time.method == :yoshida4 ? (1 / (2 - cbrt(2)), 1 - 2 / (2 - cbrt(2)), 1 / (2 - cbrt(2))) :
    (1.0,)

function validate_propagation(time, drive, absorber)
    validate(time)
    validate(drive)
    if time.method == :yoshida4
        drive isa NoDrive || throw(
            ArgumentError(
                "The implemented Yoshida composition requires a time-independent Hamiltonian",
            ),
        )
        absorber === nothing ||
            all(iszero, absorber) ||
            throw(ArgumentError("Negative Yoshida substeps do not support a CAP"))
    end
    nothing
end

"Sparse factors for Cayley or [2/2] Pade kinetic evolution."
function dense_kinetic_propagator(grid, dt, time; mass = 1.0, kinetic = nothing)
    if time.kinetic_exponential == :fft
        grid.boundary == :periodic ||
            throw(ArgumentError("FFT kinetics requires periodic boundaries"))
        phase = cis.(-dt .* Q.kinetic_eigenvalues(grid.n, grid.h, grid.fd_order; mass))
        return psi -> ifft(phase .* fft(psi, 1), 1)
    end
    kinetic === nothing && (kinetic = dense_kinetic(grid, mass))
    coefficients =
        time.kinetic_exponential == :cayley ? (0.5 + 0im,) :
        (1 / 4 + im / (4sqrt(3)), 1 / 4 - im / (4sqrt(3)))
    factors = map(
        coefficient -> lu(sparse(I, grid.n, grid.n) + im * dt * coefficient * kinetic),
        coefficients,
    )
    function apply_step(psi)
        for factor in factors
            psi = 2 .* (factor \ psi) .- psi
        end
        psi
    end
end

"Differences from a periodic binary grid, supported only at physical-domain boundaries."
function finite_difference_boundary_correction(grid::LineGrid1D; mass = 1.0)
    binary_size = 1 << grid.R
    coefficients = Q.finite_difference_coefficients(grid.fd_order)
    grid.boundary == :periodic &&
        grid.n == binary_size &&
        return (; indices = Int[], correction = zeros(0, 0))
    changes = Dict{Tuple{Int,Int},Float64}()
    function add_entry(row, column, value)
        key = (row, column)
        changes[key] = get(changes, key, 0.0) + value
    end
    radius = length(coefficients) - 1
    rows = sort!(
        unique!(
            filter(
                row -> 1 <= row <= binary_size,
                vcat(
                    collect(1:radius),
                    collect((grid.n - radius + 1):(grid.n + radius)),
                    collect((binary_size - radius + 1):binary_size),
                ),
            ),
        ),
    )
    for row in rows, offset = (-radius):radius
        iszero(offset) && continue
        value = -coefficients[abs(offset) + 1] / (2mass * grid.h^2)
        periodic_column = mod1(row + offset, binary_size)
        if row <= grid.n
            add_entry(row, periodic_column, -value)
            column = row + offset
            grid.boundary == :periodic && (column = mod1(column, grid.n))
            1 <= column <= grid.n && add_entry(row, column, value)
        elseif periodic_column <= grid.n
            add_entry(row, periodic_column, -value)
        end
    end
    filter!(entry -> !iszero(last(entry)), changes)
    indices = sort!(unique!([index for pair in keys(changes) for index in pair]))
    correction = zeros(length(indices), length(indices))
    positions = Dict(index => position for (position, index) in enumerate(indices))
    for ((row, column), value) in changes
        correction[positions[row], positions[column]] = value
    end
    (; indices, correction)
end

"Factor an already sampled one-dimensional kinetic vector with a global L2 budget."
function kinetic_vector_train(values, tolerance, cap)
    R = trailing_zeros(length(values))
    Q.with_sampling_seed(0) do
        sites = [Q.ITensors.Index(2, "kinetic=$site") for site = 1:R]
        tensor = Q.ITensors.ITensor(reshape(values, ntuple(_ -> 2, R)), sites...)
        train = Q.QTT(
            Q.ITensorMPS.MPS(tensor, sites; cutoff = tolerance^2 / max(1, R - 1), maxdim = cap),
        )
        error = norm(Q.full_values(train) - values)
        error <= max(tolerance, 100eps(Float64)) * norm(values) ||
            throw(ErrorException("Kinetic vector reconstruction exceeds its L2 budget: $error"))
        train
    end
end

"Outer product u*v^T in MPO order; the boundary inverse uses no complex conjugation."
function kinetic_outer_product(left::Q.QTT, right::Q.QTT)
    map(left.c, right.c) do a, b
        reshape(
            reshape(a, size(a, 1), 1, 2, 1, size(a, 3), 1) .*
            reshape(b, 1, size(b, 1), 1, 2, 1, size(b, 3)),
            size(a, 1) * size(b, 1),
            2,
            2,
            size(a, 3) * size(b, 3),
        )
    end
end

"Construct (I+sigma*T)^-1 from a circulant inverse and explicit boundary outer products."
function kinetic_resolvent_mpo(grid::LineGrid1D, sigma, accuracy, rec; mass = 1.0)
    binary_size = 1 << grid.R
    boundary = finite_difference_boundary_correction(grid; mass)
    tolerance, cap = accuracy.kinetic_propagator_tolerance, accuracy.kinetic_propagator_cap
    if isempty(boundary.indices) && grid.fd_order == 2 && real(inv(sigma)) >= 0
        inverse = Q.green_mpo(grid.R, grid.h * sqrt(mass), inv(sigma))
        inverse[1] .*= inv(sigma)
        return inverse
    end
    eigenvalues = Q.kinetic_eigenvalues(binary_size, grid.h, grid.fd_order; mass)
    kernel = ifft(inv.(1 .+ sigma .* eigenvalues))
    # Preserve translation structure instead of sampling a mostly diagonal inverse matrix.
    term_tolerance = tolerance / (2length(boundary.indices) + 1)
    inverse = Q.circulant(kinetic_vector_train(kernel, term_tolerance, cap))
    isempty(boundary.indices) && return inverse
    indices = boundary.indices
    left = [kernel[mod(row - column, binary_size) + 1] for row = 1:binary_size, column in indices]
    right = [kernel[mod(row - column, binary_size) + 1] for row in indices, column = 1:binary_size]
    restricted_inverse = left[indices, :]
    delta = sigma .* boundary.correction
    # B=C+P*delta*P'; B^-1=C^-1-C^-1*P*(I+delta*G)^-1*delta*P'*C^-1.
    correction = left * ((I + delta * restricted_inverse) \ delta)
    for column in eachindex(indices)
        u = kinetic_vector_train(view(correction, :, column), term_tolerance, cap)
        v = kinetic_vector_train(view(right, column, :), term_tolerance, cap)
        inverse = Q.add_mpo(inverse, kinetic_outer_product(u, v); b = -1, tol = term_tolerance, cap)
    end
    inverse
end

function qtt_kinetic_propagator(grid, dt, time, accuracy, rec; mass = 1.0)
    operators = if time.kinetic_exponential == :fft
        grid.boundary == :periodic && ispow2(grid.n) ||
            throw(ArgumentError("QTT FFT kinetics requires a complete periodic binary grid"))
        kernel = ifft(cis.(-dt .* Q.kinetic_eigenvalues(grid.n, grid.h, grid.fd_order; mass)))
        train = Q.cross(
            rec,
            grid.R,
            index -> kernel[index + 1];
            tol = accuracy.kinetic_propagator_tolerance,
            cap = accuracy.kinetic_propagator_cap,
            label = "kinetic_exponential_$dt",
            require_convergence = true,
        )
        [Q.circulant(train)]
    else
        coefficients =
            time.kinetic_exponential == :cayley ? (0.5 + 0im,) :
            (1 / 4 + im / (4sqrt(3)), 1 / 4 - im / (4sqrt(3)))
        [
            kinetic_resolvent_mpo(grid, im * dt * coefficient, accuracy, rec; mass) for
            coefficient in coefficients
        ]
    end
    (; operators, cayley = time.kinetic_exponential != :fft)
end

function lift_kinetic_propagator(propagator, axis, axes)
    (;
        operators = [axis_mpo(operator, axis, axes) for operator in propagator.operators],
        propagator.cayley,
    )
end

apply_kinetic_step(action::Function, psi, accuracy) = action(psi)
function apply_kinetic_step(propagator::NamedTuple, psi, accuracy)
    for operator in propagator.operators
        updated = Q.apply_mpo(
            operator,
            psi;
            tol = propagation_orbital_tolerance(accuracy),
            cap = accuracy.orbital_cap,
        )
        psi =
            propagator.cayley ?
            Q.reconstruct_expression(
                Q.linear_expression((updated, psi), (2, -1));
                tol = propagation_orbital_tolerance(accuracy),
                cap = accuracy.orbital_cap,
            ) : updated
    end
    psi
end

"Apply the local phase by TCI reconstruction, with the caller's orbital rank requirement."
function apply_potential_phase(
    psi::Q.QTT,
    potential,
    dt,
    accuracy,
    rec;
    label,
    support = 1 << length(psi),
)
    evaluator = Q.Evaluator(psi)
    Q.cross(
        rec,
        length(psi),
        index -> index < support ? exp(-im * dt * potential(index)) * evaluator(index) : 0.0;
        tol = propagation_orbital_tolerance(accuracy),
        cap = accuracy.orbital_cap,
        label,
    )
end
