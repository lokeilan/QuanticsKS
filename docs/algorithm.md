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
The unpolarized LSDA table interpolates XC energy density and returns its
density derivative as the XC potential. This parametrization requires unit
softening, coupling, and electron mass.

For isolated 3D systems, nuclear attraction and the Coulomb kernel use voxel
averages of `1/r`, including a finite self term. Near-field values use a
rectangular-prism integral; far-field values use Gaussian quadrature. Hartree
is a free-space block-Toeplitz convolution times `dV`. TCI samples the complete
MPO contraction environments, then fits the resulting field under its rank
budget. XC uses Slater exchange and original unpolarized PZ81 correlation.
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
| L2 | Build a common canonical tangent frame from a rotating occupied orbital. |
| L3 | Project orbitals, Green-preconditioned residuals, and history directions into that frame. |
| L4 | Remove small singular directions and contract the reduced Hamiltonian, retaining orbital-direction coupling. |
| L5 | Solve the small coefficient problem with occupied-orbital orthogonality constraints. |
| L6 | Reconstruct QTT orbitals from the correction. Clear history when Green polishing is scheduled. |
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

## Restarted orbital subspace solvers

`lanczos` applies the forward Hamiltonian to the latest basis block and
orthogonalizes the result against the retained space. `davidson` adds the
Green-preconditioned residual of the Ritz orbital with the largest residual.
`block_davidson` expands with a block of preconditioned residuals.
The two Davidson variants use the shifted kinetic inverse as their
preconditioner.

The solvers use QTT search vectors and solve a small projected Ritz problem.
The trial dimension is capped at the smaller of the grid size and
`max(16, 4b)` for `b` requested orbitals. A restart retains the current Ritz
orbitals. SCF starts a new search space for each new Hamiltonian.

Hamiltonian images and preconditioned residuals enter the same direction
selection step. It normalizes each input, removes its component in the retained
space, and uses pivoted Cholesky to select independent remainders. Ritz
reconstruction shares the basis cache across orbitals. A Gram check after
fitting determines whether orthogonality needs repair.

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
