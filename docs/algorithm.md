# Algorithm

All calculations use atomic units. Orbitals and fields are represented as QTTs;
the solver does not assemble a dense spatial Hamiltonian.

## Grid and orbital normalization

In 1D, `R` binary sites represent `N = 2^R` periodic samples at `x_j = j*L/N`.
Bits run from least to most significant. In 3D, each axis has `N = 2^R`
cell-centered samples, with sites ordered `x0,y0,z0,x1,y1,z1,...`.

The stored orbital is

$$u_{ij}=\sqrt{dV}\,\psi_i(\mathbf r_j),\qquad
\rho_j=\frac{1}{dV}\sum_i f_i|u_{ij}|^2,$$

where `f_i` is the occupation and `dV` is the grid weight: `h` in 1D and
`h_x*h_y*h_z` in 3D. Euclidean inner products therefore represent spatial
integrals. Density normalization and field integration use the same weight.

## Hamiltonian

The Kohn-Sham operator is

$$H=T+V_{\mathrm{ext}}+V_{\mathrm H}[\rho]+V_{\mathrm{xc}}[\rho].$$

Kinetic energy uses second-order central differences:

$$ (Tu)_j=\sum_a
\frac{(u_j-u_{j-\hat a})+(u_j-u_{j+\hat a})}{2m h_a^2}. $$

The 1D shifts are periodic. The 3D shifts use zero exterior samples and never
wrap between axes. The zero ghost samples lie one grid spacing beyond the edge
samples, half a voxel outside the nominal box faces.

For 1D soft-Coulomb systems, external and Hartree potentials use the same
periodic interaction kernel. Hartree is a cyclic MPO contraction times `h`.
The unpolarized LSDA table uses the correlation parametrization of
[Helbig et al., Eq. (5) and Table 1][lsda-1d]. It interpolates XC energy density and returns its
density derivative as the XC potential. This parametrization requires unit
softening, coupling, and electron mass.

For isolated 3D systems, nuclear attraction and the Coulomb kernel use voxel
averages of `1/r`, including a finite self term. Near-field values use a
rectangular-prism integral; far-field values use Gaussian quadrature. Hartree
is a free-space block-Toeplitz convolution times `dV`. TCI samples the complete
MPO contraction environments, then fits the resulting field under its rank
budget. XC uses Slater exchange and original unpolarized [Perdew-Zunger (PZ81)
correlation][pz81].
`xc="none"` retains Hartree and disables XC. Custom local-potential models are
noninteracting.

## SCF loop

1. Construct static external and interaction operators and initialize orbitals.
2. Fit the orbital density, normalize its integral to the electron number, and
   apply linear or Anderson density mixing.
3. Build the Hamiltonian from the mixed density. Keep it fixed during the
   configured inner orbital updates.
4. Round orbitals to the storage budget and check their Gram matrix. Recompute
   density from the stored orbitals. In `strict` mode, perform an output Ritz
   rotation with this density's Hamiltonian, store the rotated orbitals, and
   recompute density again.
5. Evaluate total energy, the change from the previous iteration, and the
   difference between output and mixed input densities. Apply the stopping
   criterion and audit the final state.

The total energy includes occupied kinetic energy, external energy, half the
Hartree integral, XC energy, and nuclear repulsion. The XC potential itself is
not added as an XC energy term.

## LRRAP orbital update

| Stage | Operation |
|---|---|
| L1 | Contract Rayleigh values and fit the complete residual `H*u_i - epsilon_i*u_i`. |
| L2 | Build a common tangent space from a rotating occupied orbital. |
| L3 | Project orbitals, Green-preconditioned residuals, and history directions into that tangent space. |
| L4 | Remove small singular directions and contract the reduced Hamiltonian, retaining orbital-direction coupling. |
| L5 | Solve the small coefficient problem with occupied-orbital orthogonality constraints. |
| L6 | Reconstruct QTT orbitals from the correction. Clear history when a Green update is scheduled. |
| L6G | Optionally apply `(T + mu*I)^-1` to `(mu + epsilon_i - V)*u_i` and normalize. |
| L7 | Restore orthogonality and apply the required Ritz rotation. |

Tangent projection contracts the operator directly into tangent coordinates.
The 1D Green operator has an analytic discrete periodic kernel. The 3D Green
operation uses truncated QTT conjugate gradients. It accepts a result only when
the residual of the stored state meets `green_relative_tolerance`. That residual
uses exact open shifts and an untruncated linear expression, with a QR fallback
for cancellation. Fine 3D grids can require many CG iterations and large ranks.
The Green action and its residual share one FD2 expression. Its exact shift
terms are combined before the working result is compressed.

## Lanczos and Davidson solvers

`lanczos` uses KrylovKit's [Lanczos factorization iterators][lanczos]. For
one orbital, the normalized basis vectors obey the three-term recurrence

$$
Hq_j=\beta_{j-1}q_{j-1}+\alpha_jq_j+\beta_jq_{j+1}.
$$

The Ritz problem uses the real symmetric tridiagonal matrix with diagonal
`alpha` and off-diagonal `beta`. For several orbitals, `BlockLanczosIterator`
uses the block recurrence

$$
HQ_j=Q_{j-1}B_{j-1}^{\dagger}+Q_jA_j+Q_{j+1}B_j,
$$

and a Hermitian block tridiagonal Ritz matrix. The starting block retains
independent components in degenerate eigenspaces. KrylovKit performs modified
Gram-Schmidt reorthogonalization and deflates dependent residual directions.
An invariant subspace ends expansion before a zero residual is normalized.

The `LanczosVector` adapter supplies exact QTT inner products and TT addition
with the effective working tolerance and `work_cap`. Hamiltonian actions use the same sampled
FD2 expression as the physical residual. QTT fitting and truncation make these
actions inexact, so the recurrence residual is a diagnostic, not an acceptance
criterion. The solver checks the residuals of the reconstructed orbitals
separately.

`davidson` adds the Green-preconditioned residual of the Ritz orbital with the
largest residual. `block_davidson` adds a block of preconditioned residuals.
Both use the shifted kinetic inverse. They project the new directions out of
the retained space, select independent remainders with pivoted Cholesky, and
solve a generalized Ritz problem using the overlap matrix.

The trial dimension is capped at the smaller of the grid size and
`max(16, 4b)` for `b` requested orbitals. Lanczos explicitly restarts its
factorization from the current Ritz orbital or block; each inner update
advances one recurrence step. Davidson also restarts from the current Ritz
orbitals. SCF starts a new search space for each new Hamiltonian.
Ritz reconstruction shares a basis cache across orbitals. A Gram check after
fitting determines whether the returned orbitals need orthogonality repair;
this repair does not alter the retained Lanczos basis.

## Tensor construction and rank control

TCI constructs tensors with adaptive prrLU sampling. By default, its result is
kept without a subsequent SVD. A positive `post_rounding_factor` explicitly
requests rounding at `tolerance * post_rounding_factor`.

Canonicalization uses `ITensorMPS.orthogonalize!` without truncation. Default
MPO application uses library zip-up, including its input canonicalization and
final truncation. For `d` binary sites, the amplitude tolerance `tol` maps to
local squared-weight cutoff `(tol/d)^2` and final cutoff `tol^2`. The outer
solver does not append another rounding sweep. Rank caps can still limit the
achieved accuracy. Zero tolerance and sufficient rank use an untruncated product.

Ordinary tensor rounding and addition use TCI. If TCI compression fails
numerically, ITensorMPS retries from the original cores using `truncate!`, with
cutoff `tol^2` and the same rank cap. Failure records retain the original cores
and the recovery outcome. Linear-combination fallbacks use a common QR basis
and combine amplitudes before rounding, preserving small differences between
nearly equal terms.

Field construction, working orbitals, and stored orbitals have separate rank
and tolerance controls. TCI pivot error is distinct from global approximation
error and physical residual. Independent orbital operations may run in parallel;
SCF iterations and dependent tensor sweeps remain sequential.

## Stopping and physical acceptance

`energy` stops when the total-energy change meets `energy_tolerance`. `strict`
also requires the density and Gram conditions and the independent physical audit.
The audit reconstructs density and Hamiltonian from the final stored orbitals
and checks eigen-residuals, orthogonality, charge, density consistency, and energy
consistency with separate numerical budgets. `status.json` records solver
stopping and physical acceptance independently.

## Fixed-potential spectra and propagation

The 1D `fixed_spectrum` interface defaults to Block Green. Block Green,
residual-subspace, and LRRAP updates retain band-energy stopping. The three
restarted subspace solvers check the maximum residual of all requested states
and the Gram defect against `scf.eigen_tolerance` and `scf.gram_tolerance`,
respectively. Use `solver=:block_davidson` to select a method and
`scf=SCFOptions(...)` to supply numerical controls.

The 1D `propagate` interface applies a potential half step, an FD2 Cayley kinetic
step, and a second potential half step. Each potential step uses the density of
its input orbitals; the external drive is evaluated at the time midpoint.
Orbital norms and Gram drift are recorded. The 3D implementation currently
supports ground-state SCF only.

## Terminology and notation

Names follow the physical quantity or mathematical operation. The main code
references are [Gross-Pitaevskii-TCI's numerical functions][gp-code] and
[ITensorMPS's MPS/MPO examples][itensor]. The following vocabulary connects
the implementation to its sources.

| Term | Meaning and use here | Source |
|---|---|---|
| Tensor train (TT), matrix product state (MPS), core or site tensor | A chain of tensors with contracted bond indices. `QTT.c` stores the cores; this stored field retains its name for checkpoint compatibility. | [TCI paper][tci-paper], [ITensorMPS][itensor] |
| Quantics tensor train (QTT), binary site | Binary digits encode grid coordinates. `grid.R` is bits per axis; `nsites(grid)` counts all binary sites. | [Quantics TCI][qtt-paper] |
| Bond dimension, TT rank | Size of an internal tensor index. `cap` limits that size; it is separate from an error tolerance. | [TCI paper][tci-paper], [ITensorMPS][itensor] |
| Tensor cross interpolation (TCI), pivot index sets, prrLU | Adaptive entry sampling and partially rank-revealing LU. Cached pivot indices are reused, while function values are reevaluated. | [TCI paper][tci-paper] |
| Function evaluation, batch evaluation, contraction environment | Values at selected indices and partial tensor contractions used to compute them. `BlockEvaluator` evaluates several trains; `tt_cache` is a library `TTCache`. | [TCI library][tci-library] |
| Canonical form, orthogonality center, gauge condition | Tensor representation properties. Moving the orthogonality center is distinct from rank truncation. | [ITensorMPS][itensor], [TT geometry][tt-geometry] |
| Tangent space, tangent vector, tangent projection | Variations at a fixed TT, represented by left/right orthogonal cores and gauge-constrained core variations. `TangentSpace` stores those bases and coordinate offsets. | [LRRAP, Sec. 3][lrrap-paper] |
| LRRAP | Low-rank Riemannian alternating projection. The implementation adapts the paper's common tangent-space and constrained coefficient update to Kohn-Sham SCF. | [LRRAP, Eqs. (23)-(29)][lrrap-paper] |
| Gram/overlap matrix, canonical orthogonalization | `S = X'X`; `canonical_orthogonalizer` returns retained overlap eigenvectors scaled by inverse square roots. Its columns satisfy `W'SW = I`. Orbital symmetric orthogonalization instead uses the full `S^(-1/2)`. | [PySCF orthogonalization][orthogonalization] |
| Rayleigh quotient, Ritz values/vectors, Krylov subspace, Davidson correction | Projected eigenproblems and trial-space expansion. Davidson directions use preconditioned residuals. | [SLEPc eigensolver guide][eigensolvers] |
| Lanczos recurrence, block Lanczos, reorthogonalization | Three-term scalar or block recurrence; the projected Ritz matrix is tridiagonal or block tridiagonal. Explicit restart uses the retained Ritz orbitals. | [KrylovKit factorizations][lanczos] |
| Rounding, compression, truncation | Approximation at a specified TT rank/tolerance. `compress_orbitals` also checks the Gram matrix after compression. | [TT arithmetic][tt-arithmetic] |

`psi` denotes an orbital, `rho` a density, `H` the Hamiltonian, and `V` a
potential when used in a physical expression. In the LRRAP coefficient problem,
`X` and `V` instead denote orbital and tangent-direction matrices, following
the paper: `S_xx = X'X`, `S_vx = V'X`, `H_vx = V'HX`, and `H_vv = V'HV`.
The local comment states this meaning. `c` and `C` are the corresponding
orbital and tangent coefficients. Short loop indices and factorization symbols
such as `U`, `S`, and `V` remain local to the formula they express.

`Block Green` denotes the explicit Green fixed-point update written in L6G
above. It is a descriptive project method label. The 3D inverse uses [conjugate
gradient iterations][cg] with TT compression and a separately evaluated residual.
`FD2` means second-order central finite differences; displacement operators
are named `shift_mpo`. These are spatial discretization choices, independent
of QTT encoding. Density feedback uses regularized [Anderson mixing][anderson].
Propagation uses [Strang splitting][strang] with a Cayley approximation to the kinetic
exponential; the [reference GPE paper][gp-paper] motivates the QTT split-step
organization, while this project's kinetic discretization is stated above.

[lanczos]: https://jutho.github.io/KrylovKit.jl/latest/man/implementation/
[gp-code]: https://github.com/MarcelNiedermeier/Gross-Pitaevskii-TCI/blob/main/working_example/utilities.jl
[gp-paper]: https://arxiv.org/abs/2507.04262
[itensor]: https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html
[tci-paper]: https://doi.org/10.21468/SciPostPhys.18.3.104
[qtt-paper]: https://arxiv.org/abs/2303.11819
[tci-library]: https://tensor4all.org/TensorCrossInterpolation.jl/dev/
[tt-geometry]: https://arxiv.org/abs/1407.2042
[lrrap-paper]: https://arxiv.org/abs/1811.11049
[orthogonalization]: https://pyscf.org/_modules/pyscf/scf/addons.html#canonical_orth_
[eigensolvers]: https://slepc.upv.es/release/documentation/manual/eps.html
[tt-arithmetic]: https://arxiv.org/abs/2011.06532
[anderson]: https://doi.org/10.1145/321296.321305
[lsda-1d]: https://pure.mpg.de/rest/items/item_1351618_5/component/file_1383643/content
[pz81]: https://doi.org/10.1103/PhysRevB.23.5048
[cg]: https://petsc.org/release/manualpages/KSP/KSPCG/
[strang]: https://doi.org/10.1137/0705041
