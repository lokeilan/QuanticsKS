# Algorithm

All calculations use atomic units. Orbitals and fields are represented as QTTs;
the solver does not assemble a dense spatial Hamiltonian.

## Grid and orbital normalization

In 1D, `R` binary sites represent `N = 2^R` periodic samples at `x_j = j*L/N`.
Bits run from least to most significant. In 3D, axis `a` has `N_a = 2^R_a`
cell-centered samples. A scalar `R` gives equal axis resolutions; a triple gives
independent resolutions. Sites follow `x0,y0,z0,x1,y1,z1,...`, skipping an axis
after its last bit. Box lengths and grid resolutions are independent inputs.

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
The static external potential is fitted once at `external_tolerance`; local
Hamiltonian samples reuse that QTT instead of repeating the sum over nuclei.
The independent audit rebuilds it with its stricter external budget.
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

The default 3D initial orbitals multiply a sum of nuclear exponential envelopes
by distinct box sine modes. Modes are ordered by `sum_a (mode_a/L_a)^2`, using
the actual box lengths. Nuclear grid points seed TCI sampling. The occupied
functions are normalized and orthogonalized before entering the shared SCF loop.
Custom initial orbitals can be supplied with `state_from_functions`.

1. Construct static external and interaction operators and initialize orbitals.
2. Fit the initial orbital density and normalize its integral to the electron
   number. Later iterations reuse the density already computed from the previous
   stored output orbitals. Apply linear or Anderson density mixing.
3. Build the Hamiltonian from the mixed density. Keep it fixed during the
   configured inner orbital updates.
4. Round orbitals to the storage budget and check their Gram matrix. Recompute
   density from the stored orbitals.
5. Evaluate total energy, the change from the previous iteration, and the
   difference between output and mixed input densities. Apply the stopping
   criterion and audit the final state. In `strict` mode, only an iterate that
   meets the energy, density, and Gram gates becomes a final-state candidate.
   Rotate that candidate with the output-density Hamiltonian, compress it,
   recompute its density and energy, and audit the resulting stored orbitals.
   Accept the candidate only when all conditions pass; otherwise retain the
   original iterate and search directions and continue SCF.

The total energy includes occupied kinetic energy, external energy, half the
Hartree integral, XC energy, and nuclear repulsion. The XC potential itself is
not added as an XC energy term.

In 1D, kinetic energy contracts the FD2 shift MPOs directly with each orbital.
If cancellation obscures the kinetic numerator, an exact low-rank gradient
contraction supplies it. Hartree energy contracts `rho' * K * rho` directly,
with prefactor `dV^2/2`; it does not construct a second Hartree field for energy.

## Standard Block LOBPCG at fixed H

`block_lobpcg` uses the same LOBPCGEigensolver.jl iteration for dense arrays and
QTT orbital blocks. The search space contains the current orbitals, the
preconditioned residuals, and retained history directions. Rayleigh–Ritz allows
full mixing of the old orbital columns. The inverse preconditioner is an
optional block callback; omission applies the identity. No Block Green
fixed-point update is added to this iteration.

The QTT adapter keeps orthogonalization updates as linear expressions, fits
the final orbital combinations with shared block evaluators, and evaluates
Hamiltonian actions through the existing sampled numerical operations. It
keeps orthogonalized search directions as coefficient expressions when only
their projections are needed. A shared block evaluator samples the complete
Hamiltonian action before TCI fitting; independently fitted basis actions are
not combined across a cancellation. It
does not assemble a grid-sized orbital matrix. Its Ritz problem uses the actual
search Gram. The Duersch history update uses that same metric in the
coefficient overlap; exact arithmetic recovers the library's identity-metric
formula. Physical residuals are refreshed from stored orbitals instead
of trusting linear combinations of cached Hamiltonian actions after truncation.
The final residual audit uses the independent residual-norm path and checks
the stored Gram. Inexact QTT operations can change the trajectory and do not
inherit exact-arithmetic monotonicity or convergence guarantees.

The implementation reuses [LOBPCGEigensolver.jl](https://github.com/JuliaMolSim/LOBPCGEigensolver.jl)
for the stabilized LOBPCG iteration and locking workflow, described by
[Nottoli et al.](https://arxiv.org/abs/2305.06668).
This interface solves a fixed Hamiltonian; it does not establish SCF convergence.

## LRRAP orbital update

| Stage | Operation |
|---|---|
| L1 | Contract Rayleigh values and physical residual norms. The 1D norm combines the kinetic core action with a fitted potential term. |
| L2 | Build a common tangent space from a rotating occupied orbital; adaptive runs use its working-accuracy approximation as the trial-space base. |
| L3 | Project orbitals, the Green correction from its smooth source, and history directions into that tangent space. |
| L4 | Remove small singular directions and contract the reduced Hamiltonian, retaining orbital-direction coupling. |
| L5 | Solve the small coefficient problem with occupied-orbital orthogonality constraints. |
| L6 | Reconstruct QTT orbitals from the correction. Clear history when a Green update is scheduled. |
| L6G | Optionally apply `(T + mu*I)^-1` to `(mu + epsilon_i - V)*u_i` and normalize. |
| L7 | Restore orthogonality and apply the required Ritz rotation. |

Tangent projection contracts the operator directly into tangent coordinates.
L4 reuses the orbital projections from L3 and forms the projected Hamiltonian
with one matrix multiplication after collecting its tangent action columns.
For `G=(T+mu*I)^(-1)`, L3 evaluates
`G*(H-epsilon)*psi = psi - G*((mu+epsilon-V)*psi)` directly in tangent
coordinates. This avoids fitting a large differential residual before applying
the smoothing inverse. L1 and the final audit still report the unpreconditioned
physical residual. L1 in 1D only needs its norm, so it contracts the sum of
the kinetic action and a smooth `(V-epsilon)*psi` fit without interpolating the
high-frequency kinetic component. The final audit independently fits the full
residual. The adaptive trial-space base uses the current working tolerance;
the original occupied orbitals still supply `S_xx` and `H_vx`, and their stored
accuracy is unchanged. This is a project adaptation of the common tangent-space
search, not a change to the physical Hamiltonian.

L5 retains the incoming orbital explicitly. It removes its component from the
tangent directions, then canonically orthogonalizes their remaining Gram
matrix. This preserves the old orbital as a feasible choice when a nearly
dependent direction is discarded. Adaptive runs raise the relative metric
cutoff to at least `eps(Float64)*kinetic_bound/eigen_tolerance`: the estimate
limits amplification of finite-difference projection roundoff by small overlap
eigenvalues. It is a conditioning policy; the independent physical audit still
determines acceptance. L6 reconstructs the two available QTT terms directly
in a common QR basis and checks the absolute truncation error before normalization.

For 1D residual sampling, shift differences are first
contracted into QTT cores; subtracting nearly equal point values would otherwise
inject roundoff divided by `h^2` at every sample. Cancellation-prone tangent
projections use the same gradient identity.

After the constrained coefficient solve, the Green update uses the Rayleigh
values already computed for its updated orbitals, rather than values belonging
to the incoming orbitals. For adaptive 1D residual actions, cancellation and
rounding use absolute action-error budgets. The error of each rounded core
difference is measured in amplitude space, and the final audit includes the
allocated kinetic-action error in its residual gate.

The 1D Green operator has an analytic discrete periodic kernel. The 3D Green
operation uses truncated QTT conjugate gradients. It accepts a result only when
the residual of the stored state meets `green_relative_tolerance`. That residual
uses exact open shifts and an untruncated linear expression, with a QR fallback
for cancellation. Fine 3D grids can require many CG iterations and large ranks.
The Green iteration samples the complete FD2 action and its vector updates,
without constructing the six shifted QTT states. Acceptance still contracts
the untruncated discrete residual independently of these sampled updates.

## Lanczos and Davidson solvers

`lanczos` and `block_lanczos` share KrylovKit's
[Lanczos factorization iterators][lanczos]. `lanczos` selects the scalar
recurrence for one orbital and the block recurrence for several orbitals.
`block_lanczos` selects the block recurrence even for a one-orbital block.
In the scalar case, the normalized basis vectors obey the three-term recurrence

$$
Hq_j=\beta_{j-1}q_{j-1}+\alpha_jq_j+\beta_jq_{j+1}.
$$

The Ritz problem uses the real symmetric tridiagonal matrix with diagonal
`alpha` and off-diagonal `beta`. For several orbitals, `BlockLanczosIterator`
uses the block recurrence

$$
HQ_j=Q_{j-1}B_{j-1}^{\dagger}+Q_jA_j+Q_{j+1}B_j,
$$

and a Hermitian block tridiagonal Ritz matrix. Both methods share
`lanczos_step`, Ritz reconstruction, and explicit restart. The starting block retains
independent components in degenerate eigenspaces. KrylovKit performs modified
Gram-Schmidt reorthogonalization and deflates dependent residual directions.
An invariant subspace ends expansion before a zero residual is normalized;
subsequent calls reuse the existing orbitals without refitting or restarting.

Lanczos and Davidson share a `QTTVector` adapter. Consecutive projection
updates retain a linear expression over existing QTT states. Inner products
contract its terms directly; a norm or operator action reconstructs the combined
state at the effective working tolerance and `work_cap`. A single active term
uses direct scaling; multiple terms use TCI. This avoids
constructing and compressing a separate QTT for every Gram-Schmidt subtraction.
Hamiltonian actions use the same sampled FD2 expression as the physical residual.
Within SCF, both Lanczos choices sample the frozen local potential without
fitting its combined cores first. If every initial action sample is zero,
the core-algebra fallback constructs and caches the diagonal potential MPO on
demand. Zero samples alone never establish a globally zero action.
QTT fitting and truncation make these actions inexact, so the recurrence residual is a diagnostic, not an acceptance
criterion. The solver checks the residuals of the reconstructed orbitals
separately.

`davidson` and `block_davidson` implement [generalized Davidson][davidson],
with the shifted kinetic inverse as the preconditioner:

$$r_i=H\psi_i-\epsilon_i\psi_i,\qquad
t_i=(T+\mu I)^{-1}r_i.$$

The original Davidson method uses a diagonal approximation to
`H - epsilon_i*I`; this project uses `mu = green_shift` and the kinetic inverse
for both 1D and 3D. No Jacobi-Davidson correction equation is solved.
The sign of `t_i` does not change the expanded subspace.

`davidson` adds the correction for the largest unconverged residual;
`block_davidson` adds corrections for every unconverged requested orbital.
Residuals below `eigen_tolerance` do not generate corrections. The retained
space is removed using `LinearAlgebra.pinv` of its full Gram matrix, since
QTT compression can change basis orthogonality. The Gram matrix is reused
from the previous Ritz projection while that basis is unchanged. Projected
corrections remain linear expressions during dependence screening and
KrylovKit's two-pass modified Gram-Schmidt. Directions that survive screening
are fitted, and their stored norms determine normalization. Small expression norms use an
amplitude-space QR fallback to avoid subtracting nearly equal squared norms.
The small Ritz problem also uses the full overlap matrix; canonical
orthogonalization drops overlap eigenvalues below the metric cutoffs.

KrylovKit supplies the Lanczos recurrence and the shared orthogonalization,
but has no Davidson eigensolver. The Davidson iteration remains in
`solvers.jl`; its QTT corrections, rank controls, and shared SCF interface
do not require a dense spatial vector. Small projected eigenproblems use
Julia's `LinearAlgebra.eigen`, combined states use TensorCrossInterpolation,
and the existing MPO application uses ITensorMPS.

The trial dimension is capped at the smaller of the grid size and
`max(16, 4b)` for `b` requested orbitals. Lanczos explicitly restarts its
factorization from the current Ritz orbital or block; each inner update
advances one recurrence step. Davidson also restarts from the current Ritz
orbitals. SCF starts a new search space for each new Hamiltonian.
Within SCF, Lanczos reconstructs Ritz orbitals only for a restart or the final
inner step. Fixed-potential spectra reconstruct them at every requested
physical diagnostic. A deferred step leaves the orbital Gram metric unset.
Ritz reconstruction shares a basis cache across orbitals. A Gram check after
fitting determines whether the returned orbitals need orthogonality repair;
this repair does not alter the retained Lanczos basis.

## Tensor construction and rank control

TCI constructs tensors with adaptive prrLU sampling. By default, its result is
kept without a subsequent SVD. A positive `post_rounding_factor` explicitly
requests rounding at `tolerance * post_rounding_factor`.

Sample caches belong to a fixed function. Fully cached requests reuse their
values. For a partially cached tensor slice, the missing rows and columns form
one rectangular contraction per central configuration. This bounds the number
of contractions even when cache eviction leaves many different missing-entry
patterns. Rectangles can reevaluate cached points; recorded function evaluations
include this work. Only missing entries replace cache values. Orbital transforms
share these block evaluations across their coefficient columns.
Endpoint cuts minimize the remaining contraction work within a byte budget
computed from the actual bond dimensions of the whole block. Tables are built
by matrix products without retaining a duplicate dictionary of partial
endpoints. Each active worker task owns its buffers and bounded sample/TT
caches, sharing endpoint tables and QTT cores read-only. The retained capacities
and rectangle workspace are divided among the workers. Linear combinations
and densities reduce bounded orbital rectangles immediately. This path applies
at every grid resolution; it has no switch at R16 or R18.

The 1D initial orbitals share an envelope and differ by Fourier phases. Their
orthogonalization contracts the original QTT Gram matrix, then samples each
final coefficient combination as the common envelope times its weighted phase
sum. This reuses one envelope cache without evaluating all occupied tensor
trains at every query. Per-orbital normalization, rank limits, fitting tolerance,
and the stored-orbital Gram check remain part of the shared reconstruction.

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

Optional adaptive precision starts with loose fitting budgets and tightens them
as density feedback settles. The FD2 bound `2/m * sum(1/h_axis^2)` sets the
orbital representation budget on the current grid. Precision changes reset the
energy comparison, and strict stopping is available only at the final budgets.
Optional grid continuation first solves the same physical system on coarser
grids, refines the stored orbitals, and applies the requested thresholds on the
target grid. Neither policy replaces the unfiltered physical residual with a
preconditioned residual or changes the FD2 Hamiltonian.

## Fixed-potential spectra and propagation

The 1D `fixed_spectrum` interface defaults to Block Green. Block Green,
residual-subspace, and LRRAP updates retain band-energy stopping. The
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
| LRRAP | The implementation adapts the paper's common tangent-space and constrained coefficient update to Kohn-Sham SCF. | [LRRAP, Eqs. (23)-(29)][lrrap-paper] |
| Gram/overlap matrix, canonical orthogonalization | `S = X'X`; `canonical_orthogonalizer` returns retained overlap eigenvectors scaled by inverse square roots. Its columns satisfy `W'SW = I`. Orbital symmetric orthogonalization instead uses the full `S^(-1/2)`. | [PySCF orthogonalization][orthogonalization] |
| Rayleigh quotient, Ritz values/vectors, Krylov subspace, Davidson correction | Projected eigenproblems and trial-space expansion. The Davidson methods use kinetic-inverse preconditioned residuals and belong to generalized Davidson. | [SLEPc eigensolver guide][eigensolvers], [generalized Davidson][davidson] |
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

[davidson]: https://slepc.upv.es/release/manualpages/EPS/EPSGD.html
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
