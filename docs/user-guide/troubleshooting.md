# Troubleshooting

Problems that come up when setting up and running SCHISM models, and what to do about them.

## The run returned success but there is no output

SCHISM exits with success even when it stops on an error, so rompy's `run()` cannot tell. Look at `outputs/fatal.error` and the end of `outputs/mirror.out` ([Checking a run](running.md#checking-a-run)). The most common causes are below.

## "Too few scribes"

SCHISM needs one scribe for the 2D output, one for the vertical coordinates and one per 3D output variable (two for vectors). Raise the number in the executable's argument, and the MPI processes with it (`mpirun -n 8 schism 4`), or switch off 3D output you do not need, such as `iof_hydro__26` in a 2D model ([Scribes](output.md#scribes)).

## SCHISM stops on an unknown parameter

SCHISM stops on a `param.nml` parameter it does not know. Check that [`schema_version`](configuration.md#schism-versions) matches the SCHISM version you run.

## A validation error when creating the configuration

The message gives the path to the field and what is wrong. Some that need explaining:

| Message | Cause |
|---|---|
| `Unknown field 'x'. Did you mean 'y'?` | A misspelt parameter name |
| `At least one of rough, drag, manning must be set` | SCHISM always needs a bottom friction: set one on [`SCHISMGrid`](mesh.md#bottom-friction) |
| `nml.param.opt.nchi=... makes SCHISM read a different friction file` | `nchi` does not match the grid's friction field. Leave `nchi` unset |
| `Open boundaries [...] of the mesh have no setup` | Every open boundary needs a setup: add it to `boundaries`, or set `default_boundary` ([Open boundaries](boundaries.md)) |
| `Tidal data is required for HARMONIC or HARMONICEXTERNAL boundary types` | A boundary with type 3 or 5 needs `tidal_data` |
| `schout.nhot_write (...) must be a multiple of core.ihfskip (...)` | Hotstart output must fall at the end of an output file |
| `Only SZ vertical grids (ivcor=2) can be generated` | LSC² vertical grids are made with SCHISM's `gen_vqs`; give the `vgrid.in` as a file ([Vertical grid](mesh.md#the-vertical-grid)) |
| `Hotstart generation requires both temperature and salinity sources` | `hotstart_config` reads the boundary's `temp_source` and `salt_source` ([Initial conditions](initial-conditions.md#from-an-ocean-model)) |

## WWM stops at start-up

WWM's time step, `wwminput.proc.deltc`, must equal `param.core.dt × param.opt.nstep_wwm`, and the spectral grid (`grid.msc`, `grid.mdc`) must match `core.msc2` and `core.mdc2` ([Waves](waves.md)). Run the executable built with WWM, `schism_wwm` in the Docker image.

## sflux: "num_file_times .gt. max_file_times"

SCHISM reads at most 1000 time records from each sflux file, and rompy-schism writes each source into one file. Use coarser atmospheric data, or split the simulation into [chained runs](initial-conditions.md#chained-runs) ([Atmosphere](atmosphere.md)).

## The atmospheric file is empty

The latitudes of the source must increase. Sort them with a rompy `Filter`: `filter=Filter(sort={"coords": ["latitude"]})` ([Atmosphere](atmosphere.md#wind-and-pressure)).

## Spurious currents at the open boundary

A boundary forced with the tide alone does not know about flows driven by the wind or the ocean, which then turn or pile up where it meets the coast. Move the boundary away from the area of interest, or force it with an ocean model that includes them ([From an ocean model](boundaries.md#from-an-ocean-model)). Before combining an ocean model with tides, check that its sea level is relative to the same datum as the mesh.

## The model is too slow or unstable

- **Time step.** SCHISM is semi-implicit: long time steps are usually fine, and very short ones add numerical diffusion. Aim for a Courant number above about 0.4 ([Mesh](mesh.md#checking-a-mesh-against-the-time-step)).
- **Wetting and drying.** For strong wetting and drying, raise `opt.thetai` towards 1, and check `opt.h0`.
- **Open boundary nodes** must not dry: keep the open boundary in water a few metres deep.
