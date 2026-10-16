# Configuration

A SCHISM model is a [`SCHISMConfig`][rompy_schism.config.SCHISMConfig], run by a rompy [`ModelRun`][rompy.model.ModelRun]. This page describes the fields of the configuration, how to write it in Python or YAML, and how it selects the SCHISM version.

## SCHISMConfig

| Field | Type | What it is |
|---|---|---|
| `grid` | [`SCHISMGrid`][rompy_schism.grid.SCHISMGrid] | The mesh, the vertical grid and the property files. Required. See [Mesh and vertical grid](mesh.md) |
| `data` | [`SCHISMData`][rompy_schism.data.SCHISMData] | The inputs written from data: [open boundaries](boundaries.md) and [initial conditions](initial-conditions.md), the [atmosphere](atmosphere.md) and [waves](waves.md) |
| `nml` | [`NML`][rompy_schism.namelists.schism.NML] | The namelists. See [Model settings](model-settings.md). By default, `param.nml` with SCHISM's defaults |
| `schema_version` | `"schism-v5.13"` or `"schism-v5.14"` | The SCHISM version the files are written for. See [SCHISM versions](#schism-versions) |
| `template` | path | The workspace template. The default creates the `outputs/` folder SCHISM writes to |

The `ModelRun` around it sets the run: `run_id`, `period` (a [`TimeRange`][rompy.core.time.TimeRange]) and `output_dir`. The workspace is written to `output_dir/run_id`.

## In Python

Build the parts and pass them to the configuration, as in [Your first model](../getting-started/first-model.md). Nested objects can also be given as dictionaries, which is convenient for namelist groups:

```python
from rompy_schism.namelists import NML, Param

nml = NML(param=Param(core={"dt": 120.0}, opt={"dramp": 1.0}))
```

Objects can be shared between configurations. To build variants of a model, for example for a sensitivity test, change only the parts that differ and give each `ModelRun` its own `run_id` ([Friction sensitivity](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/friction_sensitivity/) does this).

## In YAML

A YAML file holds the fields of a `ModelRun`, with the SCHISM configuration under `config`. Each nested object has the same fields as its Python class:

```yaml
run_id: tide
output_dir: runs
period:
  start: 2023-01-01T00:00
  end: 2023-01-03T00:00
  interval: 1h
config:
  model_type: schism
  grid:
    grid_type: schism
    hgrid:
      model_type: data_blob
      source: hgrid.gr3
    drag: 0.0025
  data:
    data_type: schism
    boundary_conditions:
      data_type: boundary_conditions
      tidal_data:
        tidal_database: tides
        tidal_model: TPXO9
        constituents: [M2, S2, N2, K2, K1, O1, P1, Q1]
        nodal_corrections: true
      default_boundary:
        elev_type: 3  # tidal elevation
        vel_type: 3  # tidal currents
  nml:
    param:
      core:
        ibc: 1  # barotropic
        ibtp: 0
        dt: 120.0
```

`model_type` selects the model (`schism`), and `grid_type`, `data_type` and the `model_type` of nested objects name the classes of the parts, so rompy can tell which class a block describes. Paths are relative to the folder you run from.

Load it in Python with `ModelRun(**yaml.safe_load(...))`, which checks everything, or use the `rompy` command line:

```bash
rompy validate model.yml    # check the configuration
rompy generate model.yml    # write the workspace
rompy run model.yml --backend-config docker.yml    # write and run it
```

[Tutorial 7](https://rom-py.github.io/rompy-notebooks/notebooks/schism/tutorial/07_yaml_and_cli/) builds a full hindcast as YAML and runs it from the command line.

`config.model_dump(mode="json")` gives the configuration as plain data, which can be written as YAML and loaded back. It includes every field, with the defaults, so a configuration written by hand is usually much shorter.

## SCHISM versions

SCHISM stops on a `param.nml` parameter it does not know, and parameters change between versions. `schema_version` selects the parameters of one version, for the whole configuration:

| `schema_version` | `core.nmarsh_types` | `opt.isconsv` |
|---|---|---|
| `schism-v5.13` (default) | Not accepted or written | Accepted and written |
| `schism-v5.14` | Written, 2 by default | Not written; a non-zero value is an error |

In v5.14, precipitation and evaporation are selected when SCHISM is compiled (with `PREC_EVAP`), not with `isconsv`.

```python exec="on" source="above" result="text"
from rompy.core.data import DataBlob
from rompy_schism.config import SCHISMConfig
from rompy_schism.grid import SCHISMGrid
from rompy_schism.namelists import NML, Param

grid = SCHISMGrid(hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"), drag=0.0025)
for version in ["schism-v5.13", "schism-v5.14"]:
    config = SCHISMConfig(grid=grid, schema_version=version, nml=NML(param=Param()))
    rendered = config.nml.param.render(config.schema_version)
    found = [name for name in ("nmarsh_types", "isconsv") if f"\n{name} =" in rendered]
    print(f"{version}: {', '.join(found)}")
```

Use the version you run: the [Docker image](running.md) has SCHISM v5.13.0, which matches the default.

A configuration without `schema_version` is always read as `schism-v5.13`, also in future releases that support newer versions, so existing YAML files keep their meaning. When it is written out, the version is included explicitly.
