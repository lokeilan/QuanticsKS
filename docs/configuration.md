# Configuration

## Fixed-H Block LOBPCG library interface

`block_lobpcg(H, X; preconditioner=nothing, tolerance=1e-6,
max_iterations=100, gram_tolerance=1e-8, callback=identity)` computes the lowest
`size(X,2)` eigenpairs for a dense orbital matrix. `H` is a Hermitian matrix or
a callback that applies it to an orbital block. A preconditioner callback
returns the inverse action on that block; the default is the identity.
For dense orbitals, a matrix or factorization is also accepted and applied
through a linear solve. A matrix is factorized once at entry.
The standard convergence assumptions require a Hermitian positive-definite
preconditioner and a fixed Hermitian Hamiltonian.

```julia
using QuanticsKS, LinearAlgebra, Random
H = SymTridiagonal(fill(2.0, 64), fill(-1.0, 63))
X = randn(MersenneTwister(7), 64, 4)
result = block_lobpcg(H, X;
    preconditioner = cholesky(H + I), tolerance = 1e-8)
```

For QTT, use `block_lobpcg(H, state; accuracy, tci, parallel_tasks, ...)` with
an `OrbitalState` and a fixed project Hamiltonian. The preconditioner receives
a vector of QTT residuals. The shared low-level interface also accepts
`block_lobpcg(H, orbitals, cfg, rec; work_tolerance, ...)` when the caller
already owns the numerical configuration and recorder. QTT fitting, algebra,
and rank limits retain their existing meanings. The state interface bounds
the working fit tolerance by `min(accuracy.work_tolerance, tolerance/100)`;
the final residual audit uses `accuracy.algebra_tolerance`.
The trial block can contain up to three times the orbital count. Its endpoint
tables share `tci.orbital_cache_max_bytes`; if they do not cover the whole
quantics chain, reconstruction uses TT-slice sampling. Rank and cache budgets
therefore affect the cost of fitting final block combinations.
The library requires more
grid degrees of freedom than three times the orbital count.

Results contain `orbitals`, `eigenvalues`, `residual_norms`, `gram_defect`,
`converged`, `iterations`, `residual_history`, and `elapsed_seconds`.
`residual_norms` are freshly audited; history contains the iteration's working
residuals. Callbacks receive the iteration number, eigenvalues, working
residual norms, and locked count. Convergence requires every returned orbital
residual to meet `tolerance` and the stored Gram defect to meet
`gram_tolerance`. An iteration limit does not imply convergence.

Inputs are TOML files. Unknown sections and options raise an error. Start from
[h4.toml](../inputs/h4.toml), [helium.toml](../inputs/helium.toml), or
[propagation.toml](../inputs/propagation.toml).

```sh
julia --project=. run.jl plan inputs/helium.toml
julia --project=. run.jl solve inputs/helium.toml results/helium
```

## Physical system

`[run].kind` is `scf` or `propagation`. `[grid].dimensions` defaults to `1`;
set it to `3` for an isolated Cartesian box. In 3D, `R` is either the common
number of bits per axis or a triple `[Rx, Ry, Rz]`.
Lengths are in bohr, energies in Hartree, and times in atomic units.

| Dimension | `[system].kind` | Main parameters |
|---|---|---|
| 1D | `hydrogen_chain` | `natoms`, `distance`; double occupations are assigned automatically |
| 1D | `soft_coulomb` | `positions`, `charges`, `occupations`, `softening`, `coupling`, `xc` |
| 1D | `cosine_potential` | `amplitude`, integer `mode`, `phase`, `offset`, `occupations` |
| 3D | `all_electron` | Cartesian `positions`, `charges`, double `occupations`, `xc` |

`[grid].cell` is the periodic length in 1D, or a scalar or three box lengths in
3D. Independent axis resolutions are supported for rectangular boxes.
`plan` reports the axis bits, grid shape, and spacing without starting SCF.
The 1D cosine potential is `offset + amplitude*cos(2*pi*mode*x/cell + phase)`.

The 1D interacting model accepts `xc="lsda"` or `"none"`; LSDA requires
`softening=coupling=mass=1`. The 3D model accepts `xc="pz81"` or `"none"`.
Explicit occupations allow charged isolated systems. Custom callable potentials
use `LocalPotential1D` or `LocalPotential3D` through Julia.

3D inputs support SCF only. The grid index representation permits `2 <= R <= 19`
in 3D, but computational cost usually becomes limiting much earlier. The basic
helium example is a coarse discretization, not a converged continuum result.

## Accuracy and ranks

| `[accuracy]` setting | Default | Purpose |
|---|---:|---|
| `field_tolerance` | `1e-8` | Effective-potential construction |
| `external_tolerance` | `1e-12` | Static external potential |
| `kernel_tolerance` | `1e-12` | Interaction kernel |
| `density_tolerance` | `1e-10` | Density fitting |
| `xc_energy_tolerance` | `1e-10` | XC energy-density fitting |
| `work_tolerance` | `1e-6` | Orbital update tensors |
| `storage_tolerance` | `1e-10` | Stored orbitals |
| `algebra_tolerance` | `1e-12` | Tensor algebra and Hartree construction |
| `kinetic_action_tolerance` | unset | Absolute 1D FD2 action error budget; adaptive accuracy resolves it from the physical residual target |
| `orbital_tolerance` | inherits `field_tolerance` | LRRAP update truncation; initial-envelope TCI uses one tenth of this value |
| `orthogonalization_tolerance` | `1e-11` | TCI reconstruction of orthogonalized and Ritz-rotated orbitals |
| `green_rhs_tolerance` | `1e-9` | Green right-hand-side fitting |
| `mixing_tolerance` | inherits `algebra_tolerance` | Reconstruction of mixed density |
| `storage_gram_tolerance` | `1e-7` | Gram-defect acceptance for stored orbitals |
| `orbital_cap` | `48` | Orbital rank limit |
| `work_cap` | `96` | Intermediate rank limit |
| `field_cap` | `64` | Default field rank limit |
| `adaptive` | `false` | Tighten fitting and storage budgets during SCF using the grid kinetic scale |
| `residual_error_fraction` | `0.1` | Fraction of the final orbital residual allocated to representation error |

Optional `external_cap`, `kernel_cap`, `density_cap`, `hartree_cap`,
`effective_potential_cap`, `xc_energy_cap`, and `drive_cap` override the rank
limit for the corresponding field. Unset values inherit `field_cap`.
`plan` and `config.json` show the resolved values. A static local potential is
reused only when it satisfies the effective-potential rank limit.

Set `orbital_tolerance` explicitly when relaxing `field_tolerance` independently
of orbital reconstruction. The LRRAP update uses the smaller of the working and
orbital tolerances. An unset value inherits `field_tolerance`. Density residuals use direct
contractions and have no fitting-tolerance option.

TCI tolerances measure sampled fitting errors, while SVD truncation tolerances
control tensor compression. Neither directly bounds a Hamiltonian residual.
Relax numerical budgets using measured errors in the quantity of interest;
keep SCF stopping, Gram acceptance, and final audit criteria separate. In
particular, fine-grid kinetic operators can amplify small storage errors.

`[tci].environment_cache_max_bytes` defaults to 8 MiB per tensor evaluator.
`orbital_cache_max_bytes` defaults to 128 MiB for a shared orbital block.
Endpoint cuts depend on the actual bond sizes and byte budget. The block budget
reserves space for samples, contraction caches and consumer rectangles; worker
tasks share endpoint tables and divide the remaining capacities. The budget
also reserves rank-dependent vector buffers for the template and worker tasks;
these reservations grow with the achieved ranks. Scalar consumers reduce small
orbital rectangles immediately. Contraction batches
reserve their possible prefix-cache growth before evaluation and split to fit
the remaining capacity. Partial-grid batches reuse the precomputed endpoint
tables for requested prefixes without duplicating the complete tables in a
dictionary. These budgets cover
evaluation caches and their planned work buffers, not Julia compilation, QTT
source cores, the TCI factorization, or total process memory.

With `adaptive=true`, the orbital fitting budget is capped by
`factor * residual_error_fraction * final_residual / kinetic_bound`, where
`kinetic_bound = 2/m * sum(1/h_axis^2)`. The factor starts at 100 for a cold
guess and at 1 for a supplied state, then decreases with the density residual.
Potential fitting is limited by both the orbital residual budget and energy
tolerance per electron; density and XC
energy fitting tighten against the requested total-energy tolerance. TCI ranks
grow as needed within `orbital_cap`; failed orbital
fits are rejected. Storage checks both Gram error and maximum individual
orbital representation error. The formula is a budget policy, not a proof
that a sampled TCI error bounds the physical residual; the final audit remains
required. Precision transitions reset energy comparisons and Anderson density
history, so a change of numerical accuracy cannot trigger stopping or reuse
residual coefficients from the previous accuracy.

Adaptive LRRAP also uses the larger of `metric_relative_cutoff` and
`eps(Float64)*kinetic_bound/eigen_tolerance` when resolving tangent directions
orthogonal to each incoming orbital. The old orbital remains available in the
coefficient solve. A computed cutoff of one or larger raises an error because
the grid exceeds this Float64 conditioning budget. The effective cutoff is
recorded as `coefficient_metric_cutoff`.

For the 1D residual action, adaptive precision allocates
`factor * residual_error_fraction * final_residual / 100` to kinetic-action
rounding, unless an explicit `kinetic_action_tolerance` is tighter. The two
core differences receive absolute budgets proportional to `h^2`. Each rounded
difference is compared with its unrounded amplitude expression; a missed
budget raises an error. This avoids retaining numerically insignificant ranks
just because the final residual is small. The audit adds the kinetic-action
budget to the measured residual norm before comparing it with the unchanged
physical threshold. Stored orbitals and kinetic-energy contractions are not
filtered by this policy.

`[scf].initial_spacing` optionally starts from a coarser grid, chosen from the
cell dimensions rather than a fixed R. `refinement_bits=2` limits each axis's
increase per stage. Intermediate stages use `coarse_energy_tolerance=1e-3`
and `coarse_density_tolerance=1e-2`, with a physical audit at their relaxed
thresholds. Only the final target-grid result uses the requested acceptance
criteria. Intermediate outputs have a `-grid-N` suffix, where N is the total
number of QTT sites; `grid-continuation.json` records the stages.

Adaptive LRRAP constructs its trial tangent space from the selected orbital
rounded to the current working tolerance. The stored occupied orbitals keep
their finer accuracy and supply the original overlap and Hamiltonian blocks.
This separates the search-space cost from the precision required to store a
fine-grid state. Internal 1D residual norms contract the kinetic core action
with a fitted smooth potential term; the final audit independently interpolates
the complete unfiltered residual.

The tangent-base tolerance is also limited by `green_rhs_tolerance`, so the
trial space retains the accuracy of the Green search source.
`[scf].adaptive_mixing=false` preserves fixed mixing by default. When enabled,
a density-residual increase above 1.5 times the lowest residual since the last
mixing response halves the mixing parameter
and clears Anderson history. A decrease below half the previous residual
allows a 20% increase. The parameter stays between the configured `mixing/16`
and `mixing`; physical stopping thresholds remain unchanged. Iteration logs
record the applied parameter and each adjustment.

Storage orthogonality uses `storage_gram_tolerance` and
`storage_compression_attempts`. Optional Green updates use
`green_update_interval` and `green_update_residual_threshold`. TOML inputs,
Julia constructors, solver controls, and generated records use these names.
Unknown option names are rejected.

`[audit].residual_fit_tolerance=1e-4` is a relative interpolation tolerance
for the unfiltered physical residual, not its acceptance threshold in Ha.
It is separate from `algebra_tolerance`: fitting a residual to machine-relative
precision can resolve fine XC-table structure that has no effect on the
requested residual norm. An audited residual within 10% of its acceptance
threshold is recomputed with a tenfold tighter fit. The requested physical
residual, density, charge and Gram thresholds remain unchanged.

`green_update_interval=5` schedules updates in every fifth SCF iteration;
zero disables that schedule. An orbital residual below
`green_update_residual_threshold=0.1` can also trigger an update. These controls
set the update schedule, independently of Green-solve accuracy and physical
acceptance.

`[tci].post_rounding_factor=0` keeps the prrLU result. A positive value requests
an additional SVD rounding at that fraction of the fitting tolerance.
`require_convergence=true` rejects a fit that does not meet the TCI stopping
conditions; the default is `false`. Neither option establishes physical accuracy.

For the 3D Green solve, `[physics].green_relative_tolerance` defaults to `1e-8`
and `green_max_iterations` to `300`. Failure to meet the stored-state residual
budget raises an error.

## Periodic 1D H100 profile

[h100-r16.toml](../inputs/h100-r16.toml) selects the periodic 1D soft-Coulomb
chain with 100 hydrogen atoms and 50 doubly occupied orbitals. The cell is
400 bohr with 4 bohr nuclear spacing. Change `[grid].R` for another resolution.
Each run starts directly on its target grid; adaptive accuracy is disabled.

The orbital, orthogonalization, storage and algebra tolerances are `1e-10`.
The work tolerance is `1e-4`, the absolute kinetic-action budget is `3e-6 Ha`,
and the Green RHS tolerance is `1e-6`. Field, mixing, density, XC-energy,
external and kernel tolerances are `5e-7`, `1e-6`, `5e-9`, `5e-8`, `1e-8`
and `1e-9`, respectively. One inner update uses two coefficient sweeps with
relative metric cutoff `1e-4`. A Green update is enabled every SCF step.

Orbital, work and Green RHS approximation caps are 64. Field, Hartree and
effective-potential caps are 64; density and XC-energy caps are 72; external
and kernel caps are 32 and 16. Exact temporary sums and shifts can have larger
structural bonds before approximation. Stored bonds and approximation caps
are recorded separately.

The input uses energy stopping at consecutive total-energy change below
`5e-6 Ha`, followed by a separate final-state audit. The inner eigensolver
target is `5e-4 Ha`. The reference orbital and density thresholds remain
`3e-4 Ha` and `3e-4`. An energy-stopped result can therefore have
`converged=true` with `audit.passed=false`; it must not be described as
passing all physical residual requirements. A separate independent energy
comparison can establish energy agreement, not continuum accuracy.

```sh
julia --threads=4,1 --project=. run.jl plan inputs/h100-r16.toml
julia --threads=4,1 --project=. run.jl solve inputs/h100-r16.toml results/h100-r16
```

The input uses four orbital workers and one BLAS thread. Logs, diagnostic
scripts, full research inputs and calibration reports stay in local outputs.

## SCF, propagation, and threads

| Setting | Meaning |
|---|---|
| `scf.criterion` | `energy` (default) or `strict` |
| `scf.energy_tolerance` | Total-energy change threshold; default `1e-6 Ha` |
| `scf.max_iterations` | SCF iteration limit |
| `scf.inner_steps` | Orbital updates at a fixed input density |
| `scf.solver` | `"lrrap"` (default), `"lanczos"`, `"block_lanczos"`, `"davidson"`, or `"block_davidson"` |
| `scf.mixing` | Density mixing weight |
| `scf.anderson` | Enable Anderson mixing |
| `scf.eigen_tolerance` | Inner eigensolver residual target and Ritz rotation threshold |
| `scf.orbital_residual_tolerance` | Final stored-orbital residual threshold; unset inherits `eigen_tolerance` |
| `scf.density_tolerance` | Density self-consistency threshold |
| `time.dt`, `time.steps` | Time step and number of propagation steps |
| `time.observe_every`, `time.snapshot_every` | Observation and saved-state intervals |

`strict` first checks energy change, density consistency, and stored Gram defect.
Only a candidate that meets these inexpensive gates receives an output-density
Ritz rotation and independent physical audit. The candidate replaces the stored
iteration state only if its energy, density, Gram, and audit checks all pass.
Rejected candidates leave the orbital and search-direction state unchanged;
their sampling work can still update recorder pivot caches. A final audit at an
iteration limit checks the actual returned state. Set
`orbital_residual_tolerance` explicitly to change physical acceptance without
changing the inner eigensolver decisions.

`"block_lanczos"` explicitly selects the block three-term recurrence, including
for one orbital. `"lanczos"` uses a scalar three-term recurrence for one orbital and block
Lanczos for several orbitals. Each `inner_steps` update advances one recurrence
step, with an explicit restart when the trial space reaches its size limit.
SCF reconstructs Ritz orbitals at a restart and at the final inner step;
fixed-potential spectra reconstruct them for each physical diagnostic.
The shared working tolerance and rank cap also control its QTT vector algebra.
`"davidson"` and `"block_davidson"` select generalized Davidson with the
shifted kinetic inverse, controlled by `scf.green_shift`. The first expands
with one residual correction; the second uses every unconverged requested
orbital. `scf.eigen_tolerance` determines which residuals need corrections.
Independent physical acceptance still uses the reconstructed orbitals.

`[audit]` sets independent final-state numerical budgets. `rank_multiplier`
scales both inherited and explicitly overridden field caps. `[guess]` controls
the 1D initial guess; 3D uses nuclear-envelope orbitals unless an initial state
is supplied through Julia. `[drive]` accepts `kind="none"` or
`kind="cosine_pulse"` for 1D propagation. Complete option fields are defined in
[models.jl](../src/models.jl).

Start Julia with `--threads=4,1` to allocate four worker threads. Set
`scf.parallel_tasks` or `time.parallel_tasks` to `0` to use available workers,
`1` for serial execution, or a positive concurrency limit. The default is `1`.
Use `[run].blas_threads=1` when parallelizing independent orbitals.

## Restart and output

`[run].checkpoint` is resolved relative to the input file. It imports orbitals
as a warm start; SCF mixing history starts again. A compatible `cross-state.jls`
beside the checkpoint can supply cached pivots. `[refinement]` controls
interpolation when the target grid is finer. In 3D, each axis may retain its
resolution or gain up to `max_added_bits`; at least one axis must become finer,
and no axis may become coarser. Scalar and equal-triple resolutions describe the
same grid. Refinement adjusts volume weights
and restores orbital orthogonality; it does not establish convergence.

Output paths are relative to the launch directory. The command-line output path
overrides `[run].output`, and existing output directories are rejected.

| Output | Contents |
|---|---|
| `input.toml`, `config.json` | Input and resolved model/options |
| `status.json` | Completion state, SCF stopping, and physical audit |
| `checkpoint.jls`, `snapshots/` | Stored orbitals and problem metadata |
| `cross-state.jls` | Reusable TCI pivot geometry |
| `scf.jsonl`, `lrrap.jsonl` or `eigensolver.jsonl` | Outer iterations and the selected orbital solver's inner updates |
| `workflow.json` | Configured-run elapsed time and restart information |
| `source/` | Runnable source and dependency snapshot |

`status.json.converged` means the selected stopping condition was reached;
`status.json.audit.passed` records physical acceptance. SCF exits with code `0`
when its stopping condition is met, `2` at an unconverged iteration limit, and
`1` on input or execution errors. Propagation returns `0` after completing its
configured steps.

Logged ranks distinguish inner-update output from stored orbitals. Parallel
subtask times may overlap; their sum is not wall time. Tensor storage, cumulative
allocation, and peak process memory are separate quantities.
`stages.jsonl` separates initial operator construction, Green-operator setup,
and initial orbitals, with wall time, compilation time, cumulative allocation,
and process memory observations. These stages exclude Julia process startup.

For a zero-sample fallback, `tci_calls.jsonl` records inclusive `seconds` and
separates any potential-construction time in `nested_tci_seconds`. Aggregate
TCI time counts that nested construction once, and each call retains its own ID.

`postprocess("results/helium")` writes `observables/trajectory.json`,
`density.csv`, and `bond_profiles.csv`. The 3D density output samples the Morton
index order; it is not a dense volume or a plane slice.
