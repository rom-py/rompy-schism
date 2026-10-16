# rompy-schism

rompy-schism sets up and runs the [SCHISM](https://schism-dev.github.io/schism/master/) ocean model from Python or YAML. It is the SCHISM plugin of [rompy](https://rom-py.github.io/rompy/), which provides the parts every model shares: the run period, data sources, model runs and backends.

SCHISM computes water levels, currents and, in 3D, temperature and salinity on an unstructured mesh. It reads a set of input files: the mesh, `param.nml` with the model settings, `bctides.in` for the open boundaries, and files for the boundary, atmospheric and initial data. rompy-schism describes them with validated Python objects, and writes the files for you:

- **The mesh and the files that go with it.** `hgrid.gr3`, the vertical grid, bottom friction and the other property files, as files or as constant values.
- **Open boundaries from tides and ocean models.** Tidal constituents from a tidal database, and water levels, currents, temperature and salinity from ocean model output, interpolated to the boundary nodes.
- **Forcing and initial conditions from data.** Atmospheric forcing (sflux), wave spectra for the WWM wave model, and a `hotstart.nc` from an ocean model.
- **Model settings with SCHISM's names.** `param.nml` and the namelists of the modules, with misspelt names rejected and the settings that follow from the rest of the model filled in.

```python
from rompy.core.time import TimeRange
from rompy.model import ModelRun
from rompy_schism.config import SCHISMConfig

config = SCHISMConfig(
    grid=grid,  # SCHISMGrid: hgrid.gr3, vgrid.in, friction and property files
    data=data,  # SCHISMData: bctides.in and boundary files, sflux/, hotstart.nc
    nml=nml,    # NML: param.nml and the namelists of the modules
)
run = ModelRun(
    run_id="tide",
    period=TimeRange(start="2023-01-01T00", end="2023-01-03T00", interval="1h"),
    output_dir="runs",
    config=config,
)
run()  # writes the workspace to runs/tide
```

[Your first model](getting-started/first-model.md) builds this example in full.

## Where to go next

| If you want to | Go to |
|---|---|
| Install rompy-schism and SCHISM | [Installation](getting-started/installation.md) |
| Build a model in five minutes | [Your first model](getting-started/first-model.md) |
| Learn step by step, with notebooks | [Tutorial](tutorial.md) |
| Understand how the configuration maps to SCHISM's input files | [How rompy-schism works](user-guide/how-it-works.md) |
| Set up a specific part of the model | The [User guide](user-guide/configuration.md) |
| Find the field for a `param.nml` parameter | [Namelist index](reference/namelist-index.md) |
| Look up a class | [Reference](reference/index.md) |

rompy-schism is one of the rompy model plugins, with [rompy-xbeach](https://rom-py.github.io/rompy-xbeach/) and [rompy-swan](https://rom-py.github.io/rompy-swan/). The [rompy docs](https://rom-py.github.io/rompy/) explain the ideas they share.
