# param.nml

SCHISM's main settings. [`Param`][rompy_schism.namelists.param.Param] has the three groups of `param.nml`; the [namelist index](namelist-index.md) lists every parameter.

::: rompy_schism.namelists.param.Param
    options:
      heading_level: 2

::: rompy_schism.namelists.param.Core
    options:
      heading_level: 2

::: rompy_schism.namelists.param.Opt
    options:
      heading_level: 2
      # model_type_pahm is a SCHISM parameter, not a pydantic method
      filters: ["!^_", "!^model_(?!type)"]

::: rompy_schism.namelists.param.Schout
    options:
      heading_level: 2

