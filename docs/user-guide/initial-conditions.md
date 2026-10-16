# Initial conditions and hotstart

A SCHISM run starts either from rest (a *cold start*) or from the state in a hotstart file, `hotstart.nc`. `opt.ihot` selects which.

| `opt.ihot` | Start |
|---|---|
| 0 | Cold start: water at rest, the forcing ramped up over `opt.dramp` days (1 by default) |
| 1 | From `hotstart.nc`, with the clock reset to 0: the run's own period, as usual |
| 2 | From `hotstart.nc`, continuing the clock of the run that wrote it |

## Cold start

From rest, the water level is 0 everywhere (or from `elev.ic` with `opt.ic_elev=1`) and the velocities are 0. The boundary forcing and the wind are ramped up from 0 over `opt.dramp` and `opt.drampwind` days, so the model is not shocked; leave this spin-up out of any analysis.

A barotropic model (`ibc=1`, see [Model settings](model-settings.md#barotropic-or-baroclinic)) needs nothing more. A **baroclinic** model also needs temperature and salinity everywhere at the start:

- from `temp.ic` and `salt.ic`, which vary horizontally, with `opt.flag_ic__1` and `opt.flag_ic__2` set to 1;
- from `ts.ic`, a single profile, with both set to 2;
- or from a hotstart file, which rompy-schism writes from an ocean model.

rompy-schism does not write `temp.ic`, `salt.ic` or `ts.ic`: add them to the workspace. Note that `flag_ic__1` and `flag_ic__2` default to 0 here, not to SCHISM's 1, so set them when you use these files.

## From an ocean model

[`HotstartConfig`][rompy_schism.data.HotstartConfig], in `boundary_conditions.hotstart_config`, interpolates temperature and salinity from an ocean model to every node and level of the mesh, at the time nearest the start of the run, and writes them to `hotstart.nc`. It reads the same dataset as the boundary: the setups must have a `temp_source` and a `salt_source` (see [From an ocean model](boundaries.md#from-an-ocean-model)), and `temp_var` and `salt_var` name the variables in it.

```python
from rompy_schism.data import HotstartConfig, SCHISMDataBoundaryConditions

conditions = SCHISMDataBoundaryConditions(
    tidal_data=tides,
    default_boundary=ocean,  # with temp_source and salt_source
    hotstart_config=HotstartConfig(enabled=True, temp_var="thetao", salt_var="so"),
)
```

When `hotstart_config` writes the file, [`SCHISMConfig`][rompy_schism.config.SCHISMConfig] sets `opt.ihot=1`, so SCHISM reads it. Setting `ihot=0` yourself is then an error, since SCHISM would ignore the file. Where the ocean model has land, nodes take the value of the nearest wet point, and profiles that end above the seabed are extended down.

The water level and velocities in this file are 0, so the boundary forcing is still ramped up. [Baroclinic 3D model](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/baroclinic_3d/) starts a model this way.

## Chained runs

A long simulation can be split into runs that follow on from each other, each starting from the state at the end of the previous one, without a new spin-up:

1. **Write hotstart files** in the first run: `schout.nhot=1`, and `schout.nhot_write`, the interval in time steps. It must be a multiple of `core.ihfskip`, the output file length, which rompy-schism checks.
2. **Combine them.** Each compute process writes its part of the mesh, `outputs/hotstart_<process>_<step>.nc`. `combine_hotstart7 -i <step>`, which comes with SCHISM and is in the [Docker image](running.md), merges the parts of one step into `hotstart_it=<step>.nc`.
3. **Start the next run from it.** Copy the combined file into the next run's workspace as `hotstart.nc`, and set `opt.ihot=1` and `opt.dramp=0`, so the forcing is not ramped up again.

```python
from rompy_schism.namelists import NML, Param

first = NML(param=Param(core={"dt": 120.0, "ihfskip": 720}, schout={"nhot": 1, "nhot_write": 720}))
following = NML(param=Param(core={"dt": 120.0, "ihfskip": 720}, opt={"ihot": 1, "dramp": 0.0}))
```

With `ihot=1`, each run has its own period, and rompy-schism writes its forcing for that period as usual. rompy-schism does not take a hotstart file from a previous run as an input: copy it into the generated workspace before running SCHISM. [Hotstart and chained runs](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/hotstart_and_chained_runs/) runs a chain and compares it with a continuous run and a cold start.
