# Grid

The mesh, `hgrid.gr3`, and the files SCHISM reads with it. [`SCHISMGrid`][rompy_schism.grid.SCHISMGrid] takes each file as a [`DataBlob`][rompy.core.data.DataBlob], or writes it with one of the generators below.

::: rompy_schism.grid.SCHISMGrid
    options:
      heading_level: 2

## Generators

Write a grid file in the workspace instead of copying one.

::: rompy_schism.grid.GR3Generator
    options:
      heading_level: 2

::: rompy_schism.grid.WWMBNDGR3Generator
    options:
      heading_level: 2

::: rompy_schism.grid.GridLinker
    options:
      heading_level: 2

::: rompy_schism.grid.GeneratorBase
    options:
      heading_level: 2

