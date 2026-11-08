# Three-dimensional QTT ground states

The active QuanticsKS solver now accepts isolated three-dimensional all-electron
Kohn-Sham systems and custom local potentials. The same QTT algebra, TCI recorder,
LRRAP orbital update, density mixing, output Ritz rotation, and independent SCF
audit are used by both spatial dimensions. This is a ground-state implementation;
3D real-time propagation and the 1D fixed-spectrum/low-pass interfaces are not
implemented for the new grid.

## Physical reference and parameter provenance

The reference selected from the code's `tensorchem` citation is M. V. Rakhuba and
I. V. Oseledets, *Grid-based electronic structure calculations: the tensor
decomposition approach*, J. Comput. Phys. 312 (2016),
[DOI](https://doi.org/10.1016/j.jcp.2016.02.023),
[arXiv:1508.07632](https://arxiv.org/abs/1508.07632).
This paper selection was inferred from the existing code, not confirmed by an
explicit paper identifier in the implementation request.

| Quantity | Reference / implementation |
|---|---|
| Physical Hamiltonian | All-electron kinetic energy, nuclear attraction, Hartree, unpolarized PZ81 LDA; paper Section 2 |
| Boundary model | Isolated finite system; no periodic images or neutralizing background |
| Hydrogen geometry | Simple-cubic finite clusters; Section 8.5 |
| Hydrogen separation | 2 angstrom = 3.7794522492515404 bohr |
| Example cluster | H(3 x 2 x 2), 12 electrons, 6 doubly occupied orbitals |
| Example box | Nuclear extent plus 10 bohr padding on each side; implementation choice |
| Example grid | `R=7` per axis, `128^3` points, 21 QTT sites; implementation starting point |
| Smoke input | Helium, `4^3` points in an 8-bohr box; execution check, not a paper energy comparison |
| Accuracy / stopping | Input-specific implementation choices; not claimed as the paper's settings |

PZ81 uses Slater exchange and the original unpolarized Perdew-Zunger correlation
coefficients, also present in
[Libxc's implementation](https://gitlab.com/libxc/libxc/-/blob/master/src/lda_c_pz.c).
There is no reuse of the one-dimensional soft-Coulomb XC table. `xc="none"`
disables XC while retaining Hartree. Charged isolated systems are allowed when
explicit occupations are supplied; the hydrogen-cluster constructor is neutral.

## Representation and operators

`BoxGrid3D(R, cell)` accepts a scalar or three box lengths in bohr. Each axis has
`N=2^R` cell-centered samples, `x_i=-L/2+(i+1/2)h`, with `h=L/N`.
Binary sites are ordered `x0,y0,z0,x1,y1,z1,...`, least significant bit first.
`grid_index` and `grid_indices` use zero-based integer coordinates.
The integer sampling backend limits this grid to `2 <= R <= 19`; the signed
Hartree-kernel grid uses three extra sites. This limit is unrelated to available
memory or practical solver cost.

Orbital cores store `sqrt(dV)*psi`, where `dV=h_x*h_y*h_z`. Occupations determine
the physical density `rho=sum_i occupation_i*abs2(u_i)/dV`. All density integrals
and Hartree quadrature use this same volume element.

Kinetic energy is the Cartesian FD2 sum with zero exterior samples. Each axis
has its own carry automaton; crossing an axis boundary does not wrap or spill
into another axis. The zero ghost samples lie one grid spacing beyond the edge
samples, half a voxel outside the nominal box faces. Box-size and grid
convergence must account for this discretization convention.

Hartree is a free-space block-Toeplitz convolution, represented by three binary
borrow states attached to a QTT of signed displacements. Orbitals and fields stay
in QTT form; no dense spatial Hamiltonian or FFT grid is built in the solver.
The existing bounded sample cache can use dense addressing on small grids.
Hartree samples use the library's complete MPO contraction environments and are
then fitted once by TCI under the Hartree rank budget. They do not use prefix
zip-up truncation of the uncanonicalized borrow MPO: its redundant suffix can
make prefix-local singular values a poor measure of the final field error.
The kernel is the average of `1/r` over a source voxel; its origin is a finite
voxel integral, not a fitted softening radius or an omitted self term. Nuclear
attraction is averaged over the observation voxel in the same way. Near-field
integrals use a rectangular-prism antiderivative; far-field values use tensor
Gaussian quadrature to avoid corner-subtraction cancellation. This is a
collocation/quadrature choice, not an exact reproduction of the paper's Galerkin
discretization or extrapolation procedure.

The Green operation solves `(T_FD2 + shift)*y = rhs` with truncated QTT conjugate
gradients. Acceptance evaluates the residual of the stored result as an
untruncated linear expression of exact open shifts. The amplitude-space QR
fallback resolves cancellation; neither the operator action nor this residual
norm is rounded to the CG work rank. Only a restarted search direction is
compressed. The relative residual must meet `green_relative_tolerance`, without
an extra factor-of-two allowance. An unmet budget raises an error. `green_relative_tolerance` and
`green_max_iterations` in `[physics]` control the inner solve; `work_cap` controls
its intermediate rank. This is currently the main scaling limitation: fine
grids can require many iterations and large ranks. The H12 input is a starting
configuration, not a completed fine-grid benchmark or a performance claim.
The H12 preset requires TCI convergence explicitly; a rank-limited field fit
therefore stops with an error instead of silently supplying an inaccurate field.
Its rank limits may need to be raised for the requested interpolation tolerance.

TCI construction retains its existing prrLU result by default. QR canonicalization
belongs to the consumers that need it; extra SVD rounding is not appended to
every 3D TCI fit. CG vector arithmetic and final orbital storage use their own
explicit truncation budgets.

MPO matrix elements, canonical frames, and ordinary MPO applications now use
ITensorMPS through TCI's conversion extension. The Hartree sampling path above
continues to use TCI's complete contraction environments. A local potential is
reused only when it fits the effective-potential rank budget. See the
[backend migration and 3D audit](itensormps-3d-20261108.md) for measured results.

## Running and restarting

```powershell
julia --startup-file=no --project=. run.jl plan inputs/h12-3d.toml
julia --startup-file=no --project=. run.jl solve inputs/he-3d-smoke.toml results/he-3d
julia --startup-file=no --project=. test/three_dimensional.jl
julia --startup-file=no --project=. scripts/validate_3d.jl outputs/new-3d-validation 3
```

```julia
using QuanticsKS
p = hydrogen_cluster(R=7, shape=(3,2,2))
# Inspect p and the rank/iteration budgets before a costly fine-grid solve.
result = ground_state(p; scf=SCFOptions(criterion=:strict), output="results/h12-3d")
state = load_checkpoint("results/h12-3d/checkpoint.jls"; problem=p)
observe(state)
```

`[grid].dimensions=3` selects the new model parser. Use `kind="all_electron"`
with explicit positions, charges, and double occupations for other geometries.
Checkpoints record spatial dimension, bit order, node locations, and boundary
conventions. They reject a mismatched supplied problem. Custom `LocalPotential3D`
checkpoints require the original callable model, as for the 1D interface.

`refine_state` uses tensor-product polynomial interpolation of cell-centered
samples with zero extension, corrects amplitude weights for the volume change,
and orthogonalizes the result. It provides a warm start, not a converged state or
continuation of old SCF mixing history. The default degree is five.

`scripts/validate_3d.jl` preserves a resolved input, source snapshot, checkpoint,
machine-readable measurements, and a report generated from those measurements.
It compares the QTT result with a separate small dense eigensolve/SCF calculation
and records energy stopping separately from physical acceptance. The regression
also checks the singular voxel integral with independent quadrature, the XC
derivative, axis-boundary handling, signed convolution, the discrete Green
residual, multi-orbital execution, refinement, and checkpoint identity.

`postprocess` writes sampled `x,y,z,density` rows in Morton-index order; this is
a sparse sample of the box, not a dense volume or a plane slice. Energy stopping
and physical acceptance remain separate. A small-grid agreement does not
establish continuum accuracy, adequate box size, or the paper's reported energy.
