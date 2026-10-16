# How rompy-schism works

SCHISM reads a workspace of input files: the mesh and its property files, the namelists with the model settings, `bctides.in` for the open boundaries, and files for boundary, atmospheric and initial data. rompy-schism builds these from a tree of validated objects. This page explains that tree, which parts come from rompy, and how the objects become the files.

## The configuration tree

```text
ModelRun                        rompy: run id, period, output directory; generates and runs
└── config: SCHISMConfig        rompy-schism: the SCHISM workspace
    ├── schema_version          the SCHISM version the files are written for
    ├── grid: SCHISMGrid        hgrid.gr3, vgrid.in, friction and other property files (*.gr3)
    ├── data: SCHISMData        the inputs written from data
    │   ├── boundary_conditions bctides.in, boundary files (elev2D.th.nc, ...), hotstart.nc
    │   ├── atmos               sflux/: wind, pressure, radiation, precipitation
    │   └── wave                wave spectra at the open boundary, for WWM
    └── nml: NML                the namelists
        ├── param               param.nml: CORE, OPT and SCHOUT
        ├── wwminput            wwminput.nml, for the WWM wave model
        └── sediment, ice, ...  the namelists of the other modules
```

Each object checks its own settings when it is created, and [`SCHISMConfig`][rompy_schism.config.SCHISMConfig] checks those that depend on each other.

The objects fall into three kinds:

- **The grid** writes the mesh and the files SCHISM reads with it. Each file is either copied from a file you give, as a [`DataBlob`][rompy.core.data.DataBlob], or generated: a property file with a constant value, a 2D or SZ vertical grid, links and WWM boundary flags derived from the mesh.
- **Data interfaces** write files from datasets, for the run's period and the mesh: [`SCHISMDataBoundaryConditions`][rompy_schism.data.SCHISMDataBoundaryConditions] interpolates tides and ocean models to the open boundary nodes, [`SCHISMDataSflux`][rompy_schism.data.SCHISMDataSflux] crops the atmospheric data to the mesh, and [`SCHISMDataWave`][rompy_schism.data.SCHISMDataWave] selects wave spectra along the open boundary.
- **Namelists** are SCHISM's settings, one model per namelist file and one field per parameter, with SCHISM's names. [`Param`][rompy_schism.namelists.param.Param] is `param.nml`; [`Wwminput`][rompy_schism.namelists.wwminput.Wwminput] is `wwminput.nml`.

## What comes from rompy

rompy-schism builds on [rompy](https://rom-py.github.io/rompy/), which provides the parts shared by all models:

| rompy provides | Used in rompy-schism for |
|---|---|
| [`ModelRun`][rompy.model.ModelRun] | Generating the workspace and running SCHISM |
| [`TimeRange`][rompy.core.time.TimeRange] | The run period: the start date and run length in `param.nml`, the times of WWM, and the time range of the data |
| [`BaseConfig`][rompy.core.config.BaseConfig] | The base of [`SCHISMConfig`][rompy_schism.config.SCHISMConfig], with the `template` and `checkout` fields |
| [`BaseGrid`][rompy.core.grid.BaseGrid] | The base of [`SCHISMGrid`][rompy_schism.grid.SCHISMGrid] |
| [`DataBlob`][rompy.core.data.DataBlob] | Files copied into the workspace as they are, such as `hgrid.gr3` |
| [`DataGrid`][rompy.core.data.DataGrid], [`DataBoundary`][rompy.core.boundary.DataBoundary] | The bases of the sflux sources and of [`SCHISMDataBoundary`][rompy_schism.data.SCHISMDataBoundary], with the `source`, `coords`, `filter`, `crop_data` and `buffer` fields |
| [`BoundaryWaveStation`][rompy.core.boundary.BoundaryWaveStation] | The base of [`SCHISMDataWave`][rompy_schism.data.SCHISMDataWave], which selects spectra along the open boundary |
| rompy sources, e.g. [`SourceFile`][rompy.core.source.SourceFile] | Reading data |
| Backends, e.g. [`DockerConfig`][rompy.backends.config.DockerConfig] | Running SCHISM locally, in Docker or on a cluster |
| The `rompy` command line | Generating and running models from YAML |

The [reference](../reference/index.md) lists the fields each class inherits from rompy together with its own.

## From objects to files

A namelist renders its file with `render()`. `param.nml` is written in full, with every parameter and its value, so the file shows exactly what SCHISM uses. Fields are named after SCHISM's parameters; array entries such as `iof_hydro(1)` are fields named `iof_hydro__1`:

```python exec="on" source="above" result="text"
from rompy_schism.namelists import Param

param = Param(core={"dt": 120.0}, schout={"iof_hydro__16": 1})
rendered = param.render()
print("\n".join(line for line in rendered.splitlines() if line.startswith(("&", "dt ", "iof_hydro(16)"))))
```

The grid and the data interfaces write their files into the workspace when it is generated, with `get(destdir, ...)`.

## Generating the workspace

Calling a [`ModelRun`][rompy.model.ModelRun] generates the workspace. rompy copies the template (the `outputs/` folder SCHISM writes to, and empty `sflux/` and `datasets/` folders) and calls the configuration with the run details. [`SCHISMConfig`][rompy_schism.config.SCHISMConfig] then:

1. writes the grid files: the mesh, the vertical grid and the property files;
2. calls the data interfaces, which write `bctides.in`, the boundary files, `sflux/`, the wave boundary and `hotstart.nc` for the run period;
3. sets the namelist parameters that follow from the run: the run length (`rnday`) and start date from the period, `opt.nws=2` when there is atmospheric data, and, with WWM, the times of `wwminput.nml` and the wave boundary file (`bouc.filewave`);
4. writes the namelists: `param.nml`, and `wwminput.nml` and the namelists of the other modules when they are set.

The workspace folder, `output_dir/run_id`, is ready for SCHISM to run.

## Settings filled in from the rest of the configuration

Some parameters follow from other parts of the model, so you do not set them yourself:

| Parameter | Set from | If you set it |
|---|---|---|
| `core.rnday`, `opt.start_year` … `opt.start_hour` | The run period | Replaced by the period |
| `opt.nws` | Atmospheric data: 2 with `data.atmos` | Replaced by 2, with a warning |
| `opt.nchi` | The grid's friction: 0 for `drag`, -1 for `manning`, 1 for `rough` | A different value is an error |
| `opt.ihot` | 1 when `hotstart_config` writes `hotstart.nc` | 0 is an error; 1 and 2 are kept |
| `wwminput` times | The run period | Replaced by the period |
| `wwminput.bouc.filewave` | The file written by `data.wave` | Replaced, with a warning |

## Validation

Mistakes are caught when the objects are created, before SCHISM runs:

- **Names.** Unknown parameters are rejected with a suggestion, instead of being silently ignored:

    ```python exec="on" source="above" result="text"
    from rompy_schism.namelists import Param

    try:
        Param(opt={"dramps": 2.0})
    except Exception as err:
        print(err)
    ```

- **Required parts.** The grid needs a mesh and one bottom friction, `drag`, `rough` or `manning`. Each open boundary of the mesh needs a setup, and tidal boundaries need a tidal dataset.
- **Combinations SCHISM refuses.** For example, hotstart output must be written at a multiple of the output file length, and the friction parameter must match the friction file the grid writes:

    ```python exec="on" source="above" result="text"
    from rompy.core.data import DataBlob
    from rompy_schism.config import SCHISMConfig
    from rompy_schism.grid import SCHISMGrid
    from rompy_schism.namelists import NML, Param

    grid = SCHISMGrid(hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"), manning=0.025)
    try:
        SCHISMConfig(grid=grid, nml=NML(param=Param(opt={"nchi": 0})))
    except Exception as err:
        print(err)
    ```

- **The SCHISM version.** SCHISM stops on a parameter it does not know, and parameters change between versions. `schema_version` selects the parameters of one version; see [SCHISM versions](configuration.md#schism-versions).

Some problems only show when SCHISM runs, such as too few scribes for the output. SCHISM does not report them in its exit status: see [Checking a run](running.md#checking-a-run).
