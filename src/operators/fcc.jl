# Restrict full-space potentials, kinetic energy, and Green operators to FCC blocks.

function fcc_scalar(potential, l)
    n = fcc_shape(l)
    base = vec(copy(reshape(potential, shape(l))[1:n[1], 1:n[2], :]))
    rebuilt = 2vec(fcc_expand(reshape(base, n), (1, 1)))
    norm(rebuilt - potential) <= 1e-9 * max(norm(potential), 1.0) ||
        error("input potential breaks FCC translation symmetry")
    base
end
function fcc_kernels(p, k, potential)
    B = Matrix(projector_matrix(p, k))
    (;
        t = kinetic_symbol(p, k),
        B = [fcc_restrict_matrix(p.layout, B, chi) for chi in FCC_CHARACTERS],
        potential = fcc_scalar(potential, p.layout),
    )
end
function fcc_action(p, kernels, sector, X)
    l = p.layout
    chi = FCC_CHARACTERS[sector]
    expanded = fcc_expand_matrix(l, X, chi)
    TX = similar(expanded)
    for b in axes(X, 2)
        TX[:, b] = vec(ifft(kernels.t .* fft(reshape(view(expanded, :, b), shape(l)))))
    end
    B = kernels.B[sector]
    fcc_restrict_matrix(l, TX, chi) + B * (p.projector_d .* (B' * X)) + kernels.potential .* X
end
function fcc_green(p, kernels, sector, R, mu)
    l = p.layout
    chi = FCC_CHARACTERS[sector]
    expanded = fcc_expand_matrix(l, R, chi)
    out = similar(expanded)
    for b in axes(R, 2)
        out[:, b] = vec(ifft(fft(reshape(view(expanded, :, b), shape(l))) ./ (kernels.t .+ mu)))
    end
    fcc_restrict_matrix(l, out, chi)
end
