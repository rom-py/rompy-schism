# Atmospheric forcing

SCHISM reads the atmosphere from `sflux/`: NetCDF files on the grid of the source data, which SCHISM interpolates to the mesh, and `sflux_inputs.txt`, which names the files and variables. [`SCHISMDataSflux`][rompy_schism.data.SCHISMDataSflux], in `SCHISMData.atmos`, writes them from datasets.

## The three kinds of data

| Field | Source class | Variables | Needed for |
|---|---|---|---|
| `air_1`, `air_2` | [`SfluxAir`][rompy_schism.data.SfluxAir] | Wind, sea-level pressure, air temperature, specific humidity | Wind stress and the pressure gradient; heat exchange |
| `rad_1`, `rad_2` | [`SfluxRad`][rompy_schism.data.SfluxRad] | Downward long- and short-wave radiation | Heat exchange (`opt.ihconsv=1`) |
| `prc_1`, `prc_2` | [`SfluxPrc`][rompy_schism.data.SfluxPrc] | Precipitation rate | Freshwater input |

Each kind has a primary source (`_1`), which must cover the whole mesh and run, and an optional second, usually finer one (`_2`), blended with it where it has data. The `relative_weight` of the two sources of a kind must add to 1.

## Wind and pressure

An `SfluxAir` source reads a dataset, like other rompy data sources, and names its variables with `uwind_name`, `vwind_name`, `prmsl_name`, `stmp_name` and `spfh_name`. SCHISM needs all five: those not given are filled with a standard atmosphere (no wind, 101325 Pa, 288.15 K, 0.01 kg/kg), which is fine for a model that does not compute heat exchange.

The data must have increasing latitudes. ERA5 stores them from north to south, so sort them with a rompy [`Filter`](https://rom-py.github.io/rompy/reference/filters/):

```python exec="on" source="above" result="text" session="atmos"
import tempfile
from pathlib import Path

from rompy.core.data import DataBlob
from rompy.core.filters import Filter
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.logging import config as logging_config
from rompy_schism.data import SCHISMDataSflux, SfluxAir
from rompy_schism.grid import SCHISMGrid

logging_config.update(level="ERROR")
grid = SCHISMGrid(hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"), drag=0.0025)
period = TimeRange(start="2023-01-02T00", end="2023-01-03T00", interval="1h")

atmos = SCHISMDataSflux(
    air_1=SfluxAir(
        source=SourceFile(uri="tests/data/schism/era5.nc"),
        uwind_name="u10",
        vwind_name="v10",
        prmsl_name="msl",
        filter=Filter(sort={"coords": ["latitude"]}),
    )
)
workspace = Path(tempfile.mkdtemp())
atmos.get(workspace, grid, period)
print(sorted(p.name for p in (workspace / "sflux").iterdir()))
```

The data is cropped to the mesh and to the run period plus a day on each side, as SCHISM needs. `sflux_inputs.txt` maps SCHISM's variables to the names in the file:

```python exec="on" source="above" result="text" session="atmos"
text = (workspace / "sflux" / "sflux_inputs.txt").read_text()
print("\n".join(line for line in text.splitlines() if line.startswith(("air_1", "uwind", "vwind", "prmsl", "stmp", "spfh"))))
```

## Settings

- **`opt.nws`** is SCHISM's switch for atmospheric forcing: 2 reads sflux. It is set to 2 when `data.atmos` is given, replacing any other value, with a warning.
- **`opt.wtiminc`** is the interval at which SCHISM reads the forcing, in seconds: set it to the time step of the data, for example 3600 for hourly ERA5.
- **`opt.drampwind`** ramps the wind up from zero at a cold start, over that many days.
- **Heat exchange** (`opt.ihconsv=1`, in a baroclinic model) needs the air temperature and humidity and the radiation, so give them all.

!!! warning "Long runs"
    rompy-schism writes all the records of a source into one file, and SCHISM reads at most 1000 time records from each sflux file. With hourly data, that limits a run to about 39 days (1000 hours, less the day on each side). Use coarser data, or split the simulation into [chained runs](initial-conditions.md#chained-runs).

[Tutorial 4](https://rom-py.github.io/rompy-notebooks/notebooks/schism/tutorial/04_atmospheric_forcing/) adds ERA5 wind and pressure to a tidal model and compares the two runs.
