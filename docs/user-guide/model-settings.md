# Model settings

SCHISM reads its settings from Fortran namelists. [`NML`][rompy_schism.namelists.schism.NML], in `SCHISMConfig.nml`, holds them: `param.nml`, which every run needs, and the namelists of the modules.

## param.nml

`param.nml` has three groups, and [`Param`][rompy_schism.namelists.param.Param] has one field for each:

| Group | Field | Holds |
|---|---|---|
| `&CORE` | `core`, a [`Core`][rompy_schism.namelists.param.Core] | Settings without defaults in SCHISM, which every run needs: the time step `dt`, the run length `rnday`, output intervals, barotropic or baroclinic (`ibc`) |
| `&OPT` | `opt`, an [`Opt`][rompy_schism.namelists.param.Opt] | Optional settings: start date, ramps, friction, numerics, atmospheric forcing, modules |
| `&SCHOUT` | `schout`, a [`Schout`][rompy_schism.namelists.param.Schout] | Output: which variables, how often, hotstart files, stations |

The fields have SCHISM's names and SCHISM's defaults; give only what differs, as dictionaries or as the group models. Array entries such as `iof_hydro(1)` are fields named `iof_hydro__1`. The [namelist index](../reference/namelist-index.md) lists every parameter, and the [SCHISM manual](https://schism-dev.github.io/schism/master/input-output/param.html) describes them in full.

```python exec="on" source="above" result="text"
from rompy_schism.namelists import NML, Param
from rompy_schism.namelists.param import Opt

nml = NML(
    param=Param(
        core={"ibc": 1, "ibtp": 0, "dt": 120.0},
        opt=Opt(dramp=0.5, h0=0.05),
        schout={"iof_hydro__1": 1, "iof_hydro__26": 0},
    )
)
print(nml.param.core.dt, nml.param.opt.dramp, nml.param.schout.iof_hydro__1)
```

Every parameter is written to `param.nml`, with its default when not set, so the file shows all the values SCHISM uses.

## The settings to decide first

| Setting | Default | What it does |
|---|---|---|
| `core.ibc`, `core.ibtp` | 0, 1 | Barotropic or baroclinic: see [below](#barotropic-or-baroclinic) |
| `core.dt` | 100 s | The time step. SCHISM is semi-implicit: 100 to 400 s for barotropic models, with a Courant number above about 0.4 ([Mesh](mesh.md#checking-a-mesh-against-the-time-step)) |
| `core.nspool`, `core.ihfskip` | 36, 864 | Output every `nspool` steps, in files of `ihfskip` steps ([Output](output.md)) |
| `opt.dramp`, `opt.drampwind` | 1 day | Ramps for the boundary forcing and the wind at a cold start |
| `opt.h0` | 0.01 m | Minimum depth for wetting and drying |
| `opt.thetai` | 0.6 | Implicitness, 0.5 to 1; 1 is the most stable, for strong wetting and drying |
| `opt.indvel`, `opt.ihorcon`, `opt.ishapiro` | 0, 0, 1 | Momentum stabilisation: the defaults suit most coastal models |
| `opt.itur` | 3 | Vertical turbulence closure, for 3D models |

Some parameters are set from the rest of the configuration: the run length and start date from the run period, `nws` from the atmospheric data, `nchi` from the grid's friction and `ihot` from the hotstart. [How rompy-schism works](how-it-works.md#settings-filled-in-from-the-rest-of-the-configuration) lists them.

## Barotropic or baroclinic

- **Barotropic**, `ibc=1, ibtp=0`: water levels and currents driven by tides, wind and pressure, with a constant density. Usually 2D. Set this for tidal and storm surge models.
- **Baroclinic**, `ibc=0`, the default as in SCHISM's sample `param.nml`: temperature and salinity are computed and change the density, which drives currents. It needs a 3D [vertical grid](mesh.md#the-vertical-grid), initial temperature and salinity ([Initial conditions](initial-conditions.md)), and their values at the open boundaries ([Open boundaries](boundaries.md#from-an-ocean-model)).

Set `ibc=1, ibtp=0` for a model driven by tides and wind alone: the default `ibc=0` makes it baroclinic.

## Other modules

SCHISM's modules have their own namelists, each written when its field of `NML` is set. The module must also be compiled into the SCHISM you run.

| Field | Namelist | Module |
|---|---|---|
| `wwminput` | `wwminput.nml` | WWM wave model: see [Waves](waves.md) |
| `sediment` | `sediment.nml` | SED3D sediment transport |
| `ice`, `mice` | `ice.nml`, `mice.nml` | Ice, and multi-class ice |
| `icm` | `icm.nml` | ICM water quality |
| `cosine` | `cosine.nml` | CoSiNE ecosystem |

Their parameters are on the [Modules](../reference/modules.md) and [WWM](../reference/wwminput.md) reference pages.

## SCHISM versions

Parameters change between SCHISM versions, and SCHISM stops on one it does not know. Set [`schema_version`](configuration.md#schism-versions) to the version you run.
