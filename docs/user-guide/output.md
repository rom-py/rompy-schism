# Output

SCHISM's output is set in the `&SCHOUT` and `&CORE` groups of `param.nml`: which variables, how often, and in how many files. It is written to `outputs/` in the workspace, by dedicated processes called *scribes*.

## What to write

`schout.iof_hydro(i)` switches each hydrodynamic output on (1) or off (0), with fields named `iof_hydro__i`. The ones used most, from the [SCHISM manual](https://schism-dev.github.io/schism/master/input-output/outputs.html):

| `i` | Output | Kind |
|---|---|---|
| 1 | Water level (`elevation`) | 2D |
| 2 | Air pressure | 2D |
| 13 | Bottom stress | 2D vector |
| 14 | Wind | 2D vector |
| 16 | Depth-averaged velocity (`depthAverageVelX`, `Y`) | 2D vector |
| 18, 19 | Temperature, salinity | 3D |
| 26 | Horizontal velocity (`horizontalVelX`, `Y`) | 3D vector, on by default |

Modules have their own flags, for example `iof_wwm` for waves. The [namelist index](../reference/namelist-index.md) lists them all. A 2D model does not need the 3D velocity: switch it off with `iof_hydro__26=0`.

## How often, and in how many files

- **`core.nspool`:** write every `nspool` time steps.
- **`core.ihfskip`:** start a new file (a *stack*) every `ihfskip` time steps. It must be a multiple of `nspool`.

With `dt=120` s, `nspool=30` writes every hour and `ihfskip=720` makes one file per day. SCHISM writes the 2D variables to `out2d_<stack>.nc`, and each 3D variable to its own files, such as `horizontalVelX_<stack>.nc`, with `zCoordinates_<stack>.nc` for the height of the levels.

```python
from rompy_schism.namelists import NML, Param

nml = NML(
    param=Param(
        core={"dt": 120.0, "nspool": 30, "ihfskip": 720},
        schout={"iof_hydro__1": 1, "iof_hydro__16": 1, "iof_hydro__26": 0},
    )
)
```

## Scribes

Output is written by scribes, processes that do nothing else. SCHISM needs:

- one scribe for all the 2D variables;
- one for the vertical coordinates, which are always written;
- one per 3D variable, two for a vector.

A model writing the water level, the depth-averaged velocity and the default 3D velocity needs 1 + 1 + 2 = **4 scribes**; with the 3D velocity off, 2. The number of scribes is the argument of the SCHISM executable, and is part of the number of MPI processes: `mpirun -n 6 schism 2` runs 4 compute processes and 2 scribes. With too few, SCHISM stops at start-up ([Running SCHISM](running.md)).

## Stations

SCHISM also writes time series at points, with `schout.iout_sta=1` and `schout.nspool_sta` (the interval, in time steps), at the stations listed in `station.in`. rompy-schism does not write `station.in`: add it to the workspace. Station output is plain text, `outputs/staout_<i>`, one file per variable. [Output](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/output/) shows the format of both.

## Reading the output

The `out2d` files carry the mesh with the variables, following the [UGRID](https://ugrid-conventions.github.io/ugrid-conventions/) conventions, and open with xarray:

```python
import matplotlib.tri as mtri
import xarray as xr

out2d = xr.open_mfdataset(
    "outputs/out2d_*.nc", data_vars="minimal", coords="minimal", compat="override"
)
triangulation = mtri.Triangulation(
    out2d.SCHISM_hgrid_node_x.values,
    out2d.SCHISM_hgrid_node_y.values,
    out2d.SCHISM_hgrid_face_nodes.values[:, :3] - 1,  # SCHISM numbers nodes from 1
)
```

`data_vars="minimal"` keeps the mesh variables from being repeated along time. SCHISM writes a water level at dry nodes too, so mask them with `dryFlagNode` before analysing water levels near the shore.
