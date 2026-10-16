# Boundary and initial conditions

What each open boundary does (`bctides.in`) and the files it reads, the tides, and the initial state from an ocean model (`hotstart.nc`).

::: rompy_schism.data.SCHISMDataBoundaryConditions
    options:
      heading_level: 2

::: rompy_schism.data.BoundarySetupWithSource
    options:
      heading_level: 2

::: rompy_schism.data.SCHISMDataBoundary
    options:
      heading_level: 2

::: rompy_schism.boundary_core.TidalDataset
    options:
      heading_level: 2

::: rompy_schism.data.HotstartConfig
    options:
      heading_level: 2

## Boundary types

The values of the `elev_type`, `vel_type`, `temp_type` and `salt_type` flags.

::: rompy_schism.boundary_core.ElevationType
    options:
      heading_level: 2

::: rompy_schism.boundary_core.VelocityType
    options:
      heading_level: 2

::: rompy_schism.boundary_core.TracerType
    options:
      heading_level: 2

## Shortcuts

Functions that build a [`SCHISMDataBoundaryConditions`][rompy_schism.data.SCHISMDataBoundaryConditions] for common setups.

::: rompy_schism.boundary_conditions.create_tidal_only_boundary_config
    options:
      heading_level: 2

::: rompy_schism.boundary_conditions.create_hybrid_boundary_config
    options:
      heading_level: 2

::: rompy_schism.boundary_conditions.create_river_boundary_config
    options:
      heading_level: 2

::: rompy_schism.boundary_conditions.create_nested_boundary_config
    options:
      heading_level: 2

## Base class

::: rompy_schism.tides_enhanced.BoundarySetup
    options:
      heading_level: 2

