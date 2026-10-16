# API reference

The reference is generated from the rompy-schism source code. Each class lists all its fields, including those it inherits from [rompy](https://rom-py.github.io/rompy/reference/), with their types, defaults and descriptions. The [namelist index](namelist-index.md) finds the field for a given SCHISM parameter.

| Page | What it covers |
|---|---|
| [Config](config.md) | `SCHISMConfig`, the model configuration, and the SCHISM versions it can write for |
| [Grid](grid.md) | `SCHISMGrid`: the mesh and the property files (`*.gr3`) |
| [Vertical grid](vgrid.md) | `vgrid.in`: 2D, or 3D with SZ levels |
| [Data](data.md) | `SCHISMData`, the inputs written from data |
| [Boundary and initial conditions](boundary.md) | `bctides.in`, boundary files from an ocean model, tides, and `hotstart.nc` |
| [Atmosphere](atmosphere.md) | sflux forcing |
| [Waves](waves.md) | Wave spectra at the open boundary, for WWM |
| [param.nml](param.md) | `Param` and its groups, `CORE`, `OPT` and `SCHOUT` |
| [Namelists](namelists.md) | `NML`, which holds all the namelists |
| [WWM](wwminput.md) | `wwminput.nml` |
| [Modules](modules.md) | The namelists of the sediment, ice, ICM and CoSiNE modules, and the sflux namelist |
| [Internals](internals.md) | Classes used to write the boundary and hotstart files |
| [Namelist index](namelist-index.md) | Every parameter of `param.nml` and `wwminput.nml`, and the field that sets it |
