# Configuration

Inputs are TOML files. Unknown sections and options raise an error. Start from
[h4.toml](../inputs/h4.toml), [helium.toml](../inputs/helium.toml), or
[propagation.toml](../inputs/propagation.toml).

```sh
julia --project=. run.jl plan inputs/helium.toml
julia --project=. run.jl solve inputs/helium.toml results/helium
```

## Physical system

`[run].kind` is `scf` or `propagation`. `[grid].dimensions` defaults to `1`;
set it to `3` for an isolated Cartesian box. `R` is the number of bits per axis.
Lengths are in bohr, energies in Hartree, and times in atomic units.

| Dimension | `[system].kind` | Main parameters |
|---|---|---|
| 1D | `hydrogen_chain` | `natoms`, `distance`; double occupations are assigned automatically |
| 1D | `soft_coulomb` | `positions`, `charges`, `occupations`, `softening`, `coupling`, `xc` |
| 1D | `cosine_potential` | `amplitude`, integer `mode`, `phase`, `offset`, `occupations` |
| 3D | `all_electron` | Cartesian `positions`, `charges`, double `occupations`, `xc` |
| 3D | `hydrogen_cluster` | Integer `shape`, `distance`, `padding`, `xc`; neutral hydrogen cluster |

`[grid].cell` is the periodic length in 1D, or a scalar or three box lengths in
3D. A hydrogen cluster can derive its box from nuclear extent and padding.
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
| `orbital_tolerance` | inherits `field_tolerance` | LRRAP orbital reconstruction; initial envelope uses one tenth of this value |
| `orthogonalization_tolerance` | `1e-11` | Orthogonalization and Ritz reconstruction |
| `green_rhs_tolerance` | `1e-9` | Green right-hand-side fitting |
| `mixing_tolerance` | inherits `algebra_tolerance` | Reconstruction of mixed density |
| `storage_gram_tolerance` | `1e-7` | Gram-defect acceptance for stored orbitals |
| `orbital_cap` | `48` | Orbital rank limit |
| `work_cap` | `96` | Intermediate rank limit |
| `field_cap` | `64` | Default field rank limit |

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

Storage orthogonality uses `storage_gram_tolerance` and
`storage_compression_attempts`. Optional Green updates use
`green_update_interval` and `green_update_residual_threshold`. TOML inputs,
Julia constructors, solver controls, and generated records use these names.
Unknown option names are rejected.

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

## Calibrated 1D R16 profile

[h100-r16.toml](../inputs/h100-r16.toml) specifies a periodic 1D soft-Coulomb
chain with 100 hydrogen atoms and 50 doubly occupied orbitals on an R16 grid.
These budgets are input-specific; the constructor defaults are listed above.

The profile uses `field_tolerance=1e-5`, `work_tolerance=1e-4`, and
`mixing_tolerance=1e-8`, with `work_cap=80`. Explicit field caps are 32 for the
external potential, 16 for the interaction kernel, 64 for Hartree and the
effective potential, and 128 for density and XC energy density. Orbital
reconstruction uses `orbital_tolerance=1e-8` and `orbital_cap=48`; stored
orbitals use `storage_tolerance=1e-10`.

Strict stopping uses `energy_tolerance=1e-6 Ha` and final orbital and density
thresholds of `3e-4 Ha` and `3e-4`, respectively. The inner eigensolver target
remains `eigen_tolerance=5e-5 Ha`. Passing these configured final thresholds
does not establish the tighter `5e-5` orbital or density target.

```sh
julia --threads=8,1 --project=. run.jl plan inputs/h100-r16.toml
julia --threads=8,1 --project=. run.jl solve inputs/h100-r16.toml results/h100-r16
```

The input selects available orbital workers with `parallel_tasks=0` and one
BLAS thread. The full configuration is in the input file. Logs, checkpoints,
and calibration reports remain local calculation outputs.

## SCF, propagation, and threads

| Setting | Meaning |
|---|---|
| `scf.criterion` | `energy` (default) or `strict` |
| `scf.energy_tolerance` | Total-energy change threshold; default `1e-6 Ha` |
| `scf.max_iterations` | SCF iteration limit |
| `scf.inner_steps` | Orbital updates at a fixed input density |
| `scf.solver` | `"lrrap"` (default), `"lanczos"`, `"davidson"`, or `"block_davidson"` |
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

`"lanczos"` uses a scalar three-term recurrence for one orbital and block
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
interpolation when the target grid is finer. Refinement adjusts volume weights
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

`postprocess("results/helium")` writes `observables/trajectory.json`,
`density.csv`, and `bond_profiles.csv`. The 3D density output samples the Morton
index order; it is not a dense volume or a plane slice.
