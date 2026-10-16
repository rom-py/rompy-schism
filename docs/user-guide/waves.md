# Waves with WWM

[WWM](https://schism-dev.github.io/schism/master/modules/wwm.html) is a spectral wave model that runs inside SCHISM, on the same mesh, when SCHISM is built with it. Waves and currents can be coupled both ways: the waves feel the water level and currents, and SCHISM gets the wave forces, which set up the water level in the surf zone and drive longshore currents. A model with WWM needs three things: wave spectra at the open boundary, WWM's settings, and the SCHISM executable built with WWM.

## Wave spectra at the boundary

[`SCHISMDataWave`][rompy_schism.data.SCHISMDataWave], in `SCHISMData.wave`, reads wave spectra from a wave model, selects the points nearest to the open boundary, and writes them as `<id>.nc` in the WAVEWATCH III format WWM reads. The source must give a dataset of spectra that [wavespectra](https://wavespectra.readthedocs.io) can read, such as rompy's `SourceWavespectra`:

```python
from rompy.core.source import SourceWavespectra
from rompy_schism.data import SCHISMDataWave

wave = SCHISMDataWave(
    id="wavedata",
    source=SourceWavespectra(uri="ww3-spectra.nc", reader="read_ww3"),
)
```

The spectra are cropped to the run period. `sel_method` chooses how points are selected along the boundary: the nearest spectra (the default), or an inverse-distance interpolation (`"idw"`).

## WWM's settings

WWM reads its own namelist, `wwminput.nml`, written from [`Wwminput`][rompy_schism.namelists.wwminput.Wwminput] in `NML.wwminput`. Its groups set the run (`proc`), the spectral grid (`grid`), the physics (`engs`), the numerics (`nums`), the boundary (`bouc`) and the output (`history`, `station`); the [namelist index](../reference/namelist-index.md) lists them all. rompy-schism sets:

- the start and end times of every group, from the run period;
- `bouc.filewave`, the file written by `data.wave`. A value set in the namelist is replaced, with a warning.

`wwminput.nml` is only written when `NML.wwminput` is set.

Three settings must agree with `param.nml`:

| `wwminput.nml` | `param.nml` | Rule |
|---|---|---|
| `proc.deltc`, WWM's time step (600 s by default) | `core.dt` and `opt.nstep_wwm`: WWM runs every `nstep_wwm` SCHISM steps | `deltc = dt × nstep_wwm`, or SCHISM stops |
| `grid.msc`, `grid.mdc`: frequencies and directions | `core.msc2`, `core.mdc2` | Equal |
| | `opt.icou_elfe_wwm` | The coupling: 0 none, 1 both ways; see the [reference][rompy_schism.namelists.param.Opt.icou_elfe_wwm] for the others |

```python
from rompy_schism.namelists import NML, Param
from rompy_schism.namelists.wwminput import Wwminput

nml = NML(
    param=Param(
        core={"dt": 120.0},
        opt={"icou_elfe_wwm": 1, "nstep_wwm": 5},  # WWM every 5 × 120 s
        schout={"iof_wwm__1": 1},  # significant wave height
    ),
    wwminput=Wwminput(proc={"deltc": 600}),
)
```

The wave output variables are switched on with `schout.iof_wwm(i)`, and written to the `out2d` files with the other 2D variables.

## Running

The [Docker image](running.md#with-docker) has SCHISM built with WWM as `schism_wwm`. WWM runs on the compute processes, so the number of scribes does not change:

```python
from rompy.backends import DockerConfig

backend = DockerConfig(
    image="ghcr.io/rom-py/schism:5.13.0", executable="schism_wwm 2", mpiexec="mpirun", cpu=6
)
```

The grid always writes the files WWM reads: `hgrid_WWM.gr3`, a link to the mesh, and `wwmbnd.gr3`, which marks the open boundary nodes. By default, every open boundary takes the wave spectra; [`WWMBNDGR3Generator`][rompy_schism.grid.WWMBNDGR3Generator] sets a flag per boundary.

[Waves with WWM](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/waves_wwm/) adds waves to a tide and wind model off Perth, with WAVEWATCH III spectra at the boundary.
