# Your first model

This page builds a small 2D SCHISM model of the tide off the coast of Queensland, Australia, writes its workspace and shows how to run it. Every step on this page is executed when the docs are built, so the output shown is what the code produces. The model uses the coarse test mesh and tidal data in the repository's `tests/data` folder. [Tutorial lesson 1](https://rom-py.github.io/rompy-notebooks/notebooks/schism/tutorial/01_first_model/) builds a similar model off Perth, as a notebook with maps of the results.

```python exec="on" session="first"
# Hidden setup: quiet logging and a temporary output folder.
import tempfile

from rompy.logging import config as logging_config

logging_config.update(level="ERROR")
OUT_DIR = tempfile.mkdtemp()
```

## 1. Run period

[`TimeRange`][rompy.core.time.TimeRange] comes from rompy and defines when the model runs: here two days from 1 January 2023. SCHISM takes its start date and run length from it.

```python exec="on" source="above" session="first"
from rompy.core.time import TimeRange

period = TimeRange(start="2023-01-01T00:00", end="2023-01-03T00:00", interval="1h")
```

## 2. The mesh

[`SCHISMGrid`][rompy_schism.grid.SCHISMGrid] points to the mesh, `hgrid.gr3`: the nodes with their depths, the triangles, and which parts of the outline are open sea and which are land. SCHISM also needs bottom friction everywhere; here it is a constant drag coefficient, written to `drag.gr3`. rompy-schism logs a warning that a map of friction is better.

```python exec="on" source="above" result="text" session="first"
from rompy.core.data import DataBlob
from rompy_schism.grid import SCHISMGrid

grid = SCHISMGrid(hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"), drag=0.0025)
mesh = grid.pylibs_hgrid
print(f"{mesh.np} nodes, {mesh.ne} elements, {grid.nob} open boundary")
```

No vertical grid is given, so the model is 2D: depth-averaged, with a single layer.

## 3. The tide at the open boundary

The tide enters through the open boundary. [`TidalDataset`][rompy_schism.boundary_core.TidalDataset] points to a database of tidal constituents read with [pyTMD](https://pytmd.readthedocs.io), here three constituents of the TPXO9 atlas. [`SCHISMDataBoundaryConditions`][rompy_schism.data.SCHISMDataBoundaryConditions] sets what each open boundary does: with `elev_type=3` and `vel_type=3`, the water level and currents come from the tidal constituents.

```python exec="on" source="above" session="first"
from rompy_schism.boundary_core import TidalDataset
from rompy_schism.data import (
    BoundarySetupWithSource,
    SCHISMData,
    SCHISMDataBoundaryConditions,
)

tides = TidalDataset(
    tidal_database="tests/data/schism/tides",
    tidal_model="OCEANUM-atlas",
    constituents=["M2", "S2", "N2"],
    nodal_corrections=True,
)
data = SCHISMData(
    boundary_conditions=SCHISMDataBoundaryConditions(
        tidal_data=tides,
        default_boundary=BoundarySetupWithSource(elev_type=3, vel_type=3),
    )
)
```

## 4. Model settings

`param.nml` holds SCHISM's settings in three groups, `CORE`, `OPT` and `SCHOUT`. [`NML`][rompy_schism.namelists.schism.NML]`(param=`[`Param`][rompy_schism.namelists.param.Param]`(...))` sets them, with SCHISM's own names. Only what differs from the defaults is given:

- **`ibc=1`, `ibtp=0`:** a barotropic model, driven by the tide alone. The default is a baroclinic model, which needs temperature and salinity.
- **`dt=120`:** a 2-minute time step.
- **`nspool=30`, `ihfskip=720`:** write the output every 30 steps (1 hour), in one file per 720 steps (1 day).
- **Output:** water level (`iof_hydro(1)`) and depth-averaged velocity (`iof_hydro(16)`), but not the 3D velocity, on by default (`iof_hydro(26)`).

```python exec="on" source="above" session="first"
from rompy_schism.namelists import NML, Param

nml = NML(
    param=Param(
        core={"ibc": 1, "ibtp": 0, "dt": 120.0, "nspool": 30, "ihfskip": 720},
        schout={"iof_hydro__1": 1, "iof_hydro__16": 1, "iof_hydro__26": 0},
    )
)
```

## 5. The model configuration

[`SCHISMConfig`][rompy_schism.config.SCHISMConfig] gathers the pieces. Only the grid is required.

```python exec="on" source="above" session="first"
from rompy_schism.config import SCHISMConfig

config = SCHISMConfig(grid=grid, data=data, nml=nml)
```

## 6. Generate the workspace

rompy's [`ModelRun`][rompy.model.ModelRun] combines the configuration with the run period and an output directory. Calling it writes the workspace into `output_dir/run_id`:

```python exec="on" source="above" result="text" session="first"
from pathlib import Path

from rompy.model import ModelRun

modelrun = ModelRun(run_id="first_model", period=period, output_dir=OUT_DIR, config=config)
workspace = Path(modelrun())
print(sorted(p.name for p in workspace.iterdir()))
```

Besides the mesh, the workspace has the property files SCHISM reads (`drag.gr3` and others, one value per node), `vgrid.in` and `outputs/`, where SCHISM writes. The run length and start date in `param.nml` come from the run period:

```python exec="on" source="above" result="text" session="first"
param_nml = (workspace / "param.nml").read_text()
settings = ["ibc", "dt", "rnday", "nspool", "ihfskip", "start_year", "start_month"]
print("\n".join(line for line in param_nml.splitlines() if line.split(" =")[0] in settings))
```

`bctides.in` describes the open boundaries. It starts with the constituents, their frequencies, nodal factors and phases for the start of the run; `z0` is a constant offset, the mean water level, zero here. Then comes one entry per open boundary, with its number of nodes and four flags, for elevation, velocity, temperature and salinity, followed by the amplitude and phase of each constituent at each of its nodes:

```python exec="on" source="above" result="text" session="first"
bctides = (workspace / "bctides.in").read_text().splitlines()
start = next(i for i, line in enumerate(bctides) if "!nope" in line)
print("\n".join(bctides[: start + 2]))
```

## 7. Run SCHISM

The easiest way to run SCHISM is the public Docker image, through rompy's [`DockerConfig`][rompy.backends.config.DockerConfig] backend. SCHISM always runs under MPI: `schism 2` asks for 2 *scribes*, processes that only write output, out of the 6 started.

```python
from rompy.backends import DockerConfig

backend = DockerConfig(
    image="ghcr.io/rom-py/schism:5.13.0", executable="schism 2", mpiexec="mpirun", cpu=6
)
modelrun.run(backend, workspace_dir=workspace)
```

!!! warning "Check the run"
    SCHISM returns success even when it stops with an error, so rompy cannot tell whether it finished. The run completed only if SCHISM's log, `outputs/mirror.out`, ends with "Run completed successfully"; errors are in `outputs/fatal.error`.

SCHISM writes the results to `outputs/out2d_*.nc`, one file per day here. [Running SCHISM](../user-guide/running.md) covers backends, MPI, scribes and checking a run.

## Next steps

- The [tutorial](../tutorial.md) goes through each part of the model in turn, on a model off Perth.
- [How rompy-schism works](../user-guide/how-it-works.md) explains how these objects become SCHISM's input files.
- The [User guide](../user-guide/configuration.md) covers each part of the model in detail.
