# Open boundaries and tides

Water enters and leaves a SCHISM model through its open boundaries. `bctides.in` says what SCHISM does at each one, with four flags: for the water level, the velocity, the temperature and the salinity. Some flags read their values from `bctides.in` itself (constants, tidal constituents), others from boundary files, such as `elev2D.th.nc`. [`SCHISMDataBoundaryConditions`][rompy_schism.data.SCHISMDataBoundaryConditions], in `SCHISMData.boundary_conditions`, writes both.

```python exec="on" session="bnd"
import tempfile
from pathlib import Path

from rompy.core.data import DataBlob
from rompy.core.time import TimeRange
from rompy.logging import config as logging_config
from rompy_schism.grid import SCHISMGrid

logging_config.update(level="ERROR")
OUT_DIR = Path(tempfile.mkdtemp())
grid = SCHISMGrid(hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"), drag=0.0025)
period = TimeRange(start="2023-01-01T00", end="2023-01-02T00", interval="1h")


def show_bctides(workspace):
    """Print bctides.in up to the first open boundary line."""
    lines = (workspace / "bctides.in").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "!nope" in line)
    print("\n".join(lines[: start + 2]))
```

## Setting up each boundary

[`BoundarySetupWithSource`][rompy_schism.data.BoundarySetupWithSource] sets the four flags of one open boundary, with the values or data they need. `SCHISMDataBoundaryConditions` takes one setup per open boundary:

- **`boundaries`**: a setup for each open boundary by its index in `hgrid.gr3`, from 0;
- **`default_boundary`**: the setup of the open boundaries not in `boundaries`.

Every open boundary of the mesh needs a setup, and indices the mesh does not have are an error. The flags default to 5 for the water level and velocity, and 0 for temperature and salinity, so always set the ones you need.

```python
from rompy_schism.data import BoundarySetupWithSource, SCHISMDataBoundaryConditions

conditions = SCHISMDataBoundaryConditions(
    tidal_data=tides,
    boundaries={1: BoundarySetupWithSource(elev_type=0, vel_type=2, const_flow=-500.0)},  # a river
    default_boundary=BoundarySetupWithSource(elev_type=3, vel_type=3),  # tides elsewhere
)
```

## The boundary types

The flags take SCHISM's values. The most common ones, and what rompy-schism needs for each:

| Flag | Water level, `elev_type` | Velocity, `vel_type` | Temperature, salinity, `temp_type`, `salt_type` |
|---|---|---|---|
| 0 | Not set | Not set | Not set |
| 1 | Time series, `elev.th` | Discharge time series, `flux.th` | Time series, `TEM_1.th`, `SAL_1.th` |
| 2 | Constant, `const_elev` | Constant discharge, `const_flow` | Constant, `const_temp`, `const_salt` |
| 3 | Tidal constituents, from `tidal_data` | Tidal constituents, from `tidal_data` | The initial values, for inflow |
| 4 | From a file, `elev2D.th.nc`, from `elev_source` | From a file, `uv3D.th.nc`, from `vel_source` | From files, `TEM_3D.th.nc`, `SAL_3D.th.nc`, from `temp_source`, `salt_source` |
| 5 | Tides plus the file | Tides plus the file | |
| -1 | | Flather radiation, with `mean_elev` and `mean_flow` | |
| -4 | | Relaxed to `uv3D.th.nc`, with `inflow_relax` and `outflow_relax` | |

The [enumerations](../reference/boundary.md#boundary-types) give the names of the values. The [SCHISM manual](https://schism-dev.github.io/schism/master/input-output/bctides.html) describes every option.

Discharges follow SCHISM's sign: negative into the domain. Temperature and salinity at a boundary with flags 1 to 4 are nudged towards the boundary values, by `temp_nudge` and `salt_nudge` (1, the strongest, by default).

!!! note "Files rompy-schism does not write"
    rompy-schism writes the files of types 4 and 5 from data sources. It does not write the time series of type 1 (`elev.th`, `flux.th`, `TEM_1.th`, `SAL_1.th`); add them to the workspace yourself. A type 4 or 5 setup without a source logs a warning and expects the file to be there too.

## Tides

[`TidalDataset`][rompy_schism.boundary_core.TidalDataset], in `tidal_data`, selects the tidal constituents of boundaries with type 3 or 5. They are read with [pyTMD](https://pytmd.readthedocs.io) from a tidal database: a folder with the model files and a `database.json` that says where they are and in which format. Global models such as [TPXO](https://www.tpxo.net) and [FES](https://www.aviso.altimetry.fr) work this way.

| Field | Default | What it does |
|---|---|---|
| `tidal_database`, `tidal_model` | pyTMD's default folder, `FES2014` | The database folder and the model in it |
| `constituents` | the main eight | A list, or `"major"` (M2, S2, N2, K2, K1, O1, P1, Q1) or `"all"` |
| `nodal_corrections` | off | Correct each constituent's amplitude and phase for the 18.6-year lunar nodal cycle, for the run's dates |
| `tidal_potential`, `cutoff_depth` | off, 50 m | Also apply the tide-generating force inside the domain, where it is deeper than `cutoff_depth`. Matters for large domains |
| `extrapolate_tides`, `extrapolation_distance` | off, 50 km | Take the nearest wet value of the tide model for boundary nodes where it has land |
| `mean_dynamic_topography` | 0.0 | A mean water level at the boundary, written as a constituent `z0`: a constant, or a dataset that varies along the boundary |

```python exec="on" source="above" result="text" session="bnd"
from rompy_schism.boundary_core import TidalDataset
from rompy_schism.data import BoundarySetupWithSource, SCHISMDataBoundaryConditions

tides = TidalDataset(
    tidal_database="tests/data/schism/tides",
    tidal_model="OCEANUM-atlas",
    constituents=["M2", "S2", "N2"],
    nodal_corrections=True,
)
tidal = SCHISMDataBoundaryConditions(
    tidal_data=tides, default_boundary=BoundarySetupWithSource(elev_type=3, vel_type=3)
)
tidal.get(OUT_DIR / "tidal", grid, period)
show_bctides(OUT_DIR / "tidal")
```

`bctides.in` starts with the tidal potential (none here), then the constituents with their frequency, nodal factor and phase for the start of the run, then the open boundaries. [Tidal boundaries](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/tidal_boundaries/) shows what each option changes.

## From an ocean model

[`SCHISMDataBoundary`][rompy_schism.data.SCHISMDataBoundary] interpolates a dataset to the open boundary nodes, for the run period, and writes it as the boundary file of the type that uses it. Give it to the setup as `elev_source`, `vel_source`, `temp_source` or `salt_source`, with the variables to read and the names of the dataset's coordinates:

| Source | Variables | File |
|---|---|---|
| `elev_source` | the sea level | `elev2D.th.nc` |
| `vel_source` | the eastward and northward velocity | `uv3D.th.nc` |
| `temp_source`, `salt_source` | the temperature, the salinity | `TEM_3D.th.nc`, `SAL_3D.th.nc` |

Here, the tide plus the sea level of an ocean model (`elev_type=5`), which has no tide of its own:

```python exec="on" source="above" result="text" session="bnd"
import xarray as xr
from rompy.core.source import SourceFile
from rompy_schism.data import SCHISMDataBoundary

sea_level = SCHISMDataBoundary(
    source=SourceFile(uri="tests/data/schism/hycom.nc"),
    variables=["surf_el"],
    coords={"t": "time", "x": "xlon", "y": "ylat"},
)
ocean = SCHISMDataBoundaryConditions(
    tidal_data=tides,
    default_boundary=BoundarySetupWithSource(elev_type=5, vel_type=3, elev_source=sea_level),
)
ocean.get(OUT_DIR / "ocean", grid, period)
elev2d = xr.open_dataset(OUT_DIR / "ocean" / "elev2D.th.nc")
print(dict(elev2d.time_series.sizes))
print(f"time step: {float(elev2d.time_step[0]):.0f} s")
```

The file has one value per open boundary node and time, at the dataset's time step. Use `elev_type=4` instead when the ocean model includes the tide, or there is none.

- **3D data.** For velocity, temperature and salinity in a 3D model, give the depth coordinate too (`"z": "depth"`): the data is also interpolated to SCHISM's levels at each boundary node. Profiles that end above SCHISM's deepest level are extended down from their deepest value.
- **Missing values.** Boundary nodes where the dataset has land take the value of the nearest node with data.
- **Datums.** An ocean model's sea level is relative to its own reference, which is usually not the datum of the mesh depths. Check before combining it with tides, or remove its mean ([Ocean boundaries](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/ocean_boundaries/) shows the effect).
- **Several boundaries.** Each file holds the nodes of all the open boundaries that use it, in mesh order, so boundaries that use the same file must use the same source.

A boundary file prepared elsewhere can be given as a [`DataBlob`][rompy.core.data.DataBlob], which is copied as it is.

## Rivers and other constant boundaries

A river is an open boundary with a discharge, `vel_type=2` and a negative `const_flow` in m³/s. In a baroclinic model, give its temperature and salinity too, for example as constants (type 2, with `const_temp` and `const_salt`).

```python exec="on" source="above" result="text" session="bnd"
river = SCHISMDataBoundaryConditions(
    default_boundary=BoundarySetupWithSource(
        elev_type=0, vel_type=2, const_flow=-100.0
    )
)
river.get(OUT_DIR / "river", grid, period)
show_bctides(OUT_DIR / "river")
```

## Shortcuts

Functions in `rompy_schism.boundary_conditions` build a `SCHISMDataBoundaryConditions` for common setups: [`create_tidal_only_boundary_config`][rompy_schism.boundary_conditions.create_tidal_only_boundary_config], [`create_hybrid_boundary_config`][rompy_schism.boundary_conditions.create_hybrid_boundary_config] (tides plus an ocean model), [`create_river_boundary_config`][rompy_schism.boundary_conditions.create_river_boundary_config] and [`create_nested_boundary_config`][rompy_schism.boundary_conditions.create_nested_boundary_config] (velocities relaxed to a parent model). Note that their defaults differ from `TidalDataset`'s: nodal corrections and the tidal potential are on.

`setup_type` (`"tidal"`, `"hybrid"`, `"river"` or `"nested"`) only adds checks for that kind of setup, such as requiring a tidal dataset; it does not change the boundaries.
