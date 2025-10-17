# SCHISM

## Grids

::: rompy_schism.grid.SCHISMGrid

## Data

Supporting objects for SCHISM data files.jects

::: rompy_schism.data.SfluxSource
::: rompy_schism.data.SfluxAir
::: rompy_schism.data.SfluxRad
::: rompy_schism.data.SfluxPrc
::: rompy_schism.data.SCHISMDataBoundary

Main objects

::: rompy_schism.data.SCHISMDataSflux
::: rompy_schism.data.SCHISMDataWave
::: rompy_schism.data.SCHISMDataBoundaryConditions
::: rompy_schism.data.HotstartConfig
::: rompy_schism.data.SCHISMData

## Boundary Conditions

The boundary conditions module provides a unified interface for configuring all types of SCHISM boundary conditions including tidal, ocean, river, and nested model boundaries.

[Boundary Conditions](boundary_conditions.md)

## Hotstart Configuration

The hotstart system provides integrated initial condition file generation, allowing you to create hotstart.nc files from the same ocean data sources used for boundary conditions.

[Hotstart](hotstart.md)

## Backend Framework

The backend framework provides unified execution of SCHISM simulations using Docker containers with automatic image building and comprehensive testing capabilities.

[Backend Framework](backend_framework.md)
[Backend Tutorial](backend_tutorial.md)

::: rompy_schism.data.SCHISMDataBoundaryConditions
::: rompy_schism.data.BoundarySetupWithSource
::: rompy_schism.boundary_conditions.create_tidal_only_boundary_config
::: rompy_schism.boundary_conditions.create_hybrid_boundary_config
::: rompy_schism.boundary_conditions.create_river_boundary_config
::: rompy_schism.boundary_conditions.create_nested_boundary_config

## Config Minimal

This object has been implemented to meet the initial operational requirements of CSIRO. It is likely that this will be superceded by the full implementation.

::: rompy_schism.config.SchismCSIROConfig

## Full Namelist Implementation

This object implements a set of models for each namelist and assembles a config object using this group of models.  This is curently only partly implemented.

### PARAM

::: rompy_schism.namelists.param.Core
::: rompy_schism.namelists.param.Opt
::: rompy_schism.namelists.param.Schout
::: rompy_schism.namelists.param.Vertical
::: rompy_schism.namelists.param.Param

### ICE

::: rompy_schism.namelists.ice.Ice_in
::: rompy_schism.namelists.ice.Ice

### MICE

::: rompy_schism.namelists.mice.Mice_in
::: rompy_schism.namelists.mice.Mice

### ICM

::: rompy_schism.namelists.icm.Bag
::: rompy_schism.namelists.icm.Core
::: rompy_schism.namelists.icm.Ero
::: rompy_schism.namelists.icm.Marco
::: rompy_schism.namelists.icm.Ph_icm
::: rompy_schism.namelists.icm.Poc
::: rompy_schism.namelists.icm.Sav
::: rompy_schism.namelists.icm.Sfm
::: rompy_schism.namelists.icm.Silica
::: rompy_schism.namelists.icm.Stem
::: rompy_schism.namelists.icm.Veg
::: rompy_schism.namelists.icm.Zb
::: rompy_schism.namelists.icm.Icm

### SEDIMENT

::: rompy_schism.namelists.sediment.Sed_opt
::: rompy_schism.namelists.sediment.Sed_core
::: rompy_schism.namelists.sediment.Sediment

### COSINE

::: rompy_schism.namelists.cosine.Core
::: rompy_schism.namelists.cosine.Marco
::: rompy_schism.namelists.cosine.Misc
::: rompy_schism.namelists.cosine.Cosine

### WWMINPUT

::: rompy_schism.namelists.wwminput.Coupl
::: rompy_schism.namelists.wwminput.Engs
::: rompy_schism.namelists.wwminput.Grid
::: rompy_schism.namelists.wwminput.History
::: rompy_schism.namelists.wwminput.Hotfile
::: rompy_schism.namelists.wwminput.Init
::: rompy_schism.namelists.wwminput.Nesting
::: rompy_schism.namelists.wwminput.Nums
::: rompy_schism.namelists.wwminput.Petscoptions
::: rompy_schism.namelists.wwminput.Proc
::: rompy_schism.namelists.wwminput.Station
::: rompy_schism.namelists.wwminput.Wwminput

### NML

This is the full namelist object that is the container for all the other namelist objects.

::: rompy_schism.namelists.NML

## Config Object

::: rompy_schism.config.SCHISMConfig