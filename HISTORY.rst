=======
History
=======

Relocatable Ocean Modelling in PYthon (rompy) is a modular Python library that
aims to streamline the setup, configuration, execution, and analysis of coastal
ocean models. Rompy combines templated model configuration with xarray-based
data handling and pydantic validation, enabling users to efficiently generate
model control files and input datasets for a variety of ocean and wave models.
The architecture centers on high-level execution control (`ModelRun`) and
flexible configuration objects, supporting both persistent scientific model
state and runtime backend selection. Rompy provides unified interfaces for
grids, data sources, boundary conditions, and spectra, with extensible plugin
support for new models and execution environments. Comprehensive documentation,
example Jupyter notebooks, and a robust logging/formatting framework make rompy
accessible for both research and operational workflows. Current model support
includes SWAN and SCHISM, with ongoing development for additional models and
cloud/HPC backends.

Key Features: - Modular architecture with clear separation of configuration and
execution logic - Templated, reproducible model configuration using pydantic
and xarray - Unified interfaces for grids, data, boundaries, and spectra -
Extensible plugin system for models, data sources, backends, and postprocessors
- Robust logging and formatting for consistent output and diagnostics - Example
  notebooks and comprehensive documentation for rapid onboarding - Support for
  local, Docker, and HPC execution backends

rompy is under active development—features, model support, and documentation
are continually evolving. Contributions and feedback are welcome!


********
Releases
********

Unreleased
__________

New Features
------------
* ``SCHISMDataBoundaryConditions.default_boundary`` sets up the open boundaries not listed in ``boundaries``. The factory functions use it, so ``create_tidal_only_boundary_config`` applies tidal elevation and currents to every open boundary, as documented.

Bug Fixes
---------
* 3D models work with a vertical grid generated from ``VGrid`` or ``VgridGenerator``, not only with a ``vgrid.in`` file. ``SCHISMGrid.is_3d`` was False and ``pylibs_vgrid`` failed for them, so boundary data and hotstart files were written as 2D.
* ``SCHISMGrid.is_3d`` and ``nvrt`` come from the vertical grid: a 2D ``vgrid.in`` file is no longer taken as 3D, and a 2D grid has ``nvrt=2`` instead of ``None``.
* LSC2 vertical grids, which cannot be generated here (they need SCHISM's ``gen_vqs``), are rejected when configured with a message to give ``vgrid.in`` as a file. ``VGrid()`` defaulted to LSC2 and always failed; it now defaults to SZ. ``VgridGenerator.vgrid_type`` accepts only ``2d``, ``sz`` and ``lsc2`` instead of falling back to LSC2 for other values.
* 3D boundary files (``TEM_3D.th.nc``, ``SAL_3D.th.nc``, ``uv3D.th.nc``) are interpolated to the vertical grid's own levels. They held the value extrapolated from the top of the source profile at every level.
* Source profiles are extended below the ocean model's seabed before they are interpolated to the boundary nodes and the hotstart, so profiles near the seabed are not cut short or mixed with values from elsewhere.
* The hotstart takes the source time closest to the start of the run; it always took the first time in the source.
* Boundary data missing from the source (open boundary nodes outside its wet cells, or levels below its bottom) are filled from the nearest valid data: up the water column, then from the nearest boundary node, then in time. Values at the ends of the boundary were filled with one constant, the median of all boundary values, which is often the case where an open boundary meets the coast. A warning gives the number of values filled, and a boundary without any valid data is an error.
* sflux air variables missing from the source are filled with a standard atmosphere (101325 Pa, 288.15 K, 0.01 kg/kg) instead of -999. With heat exchange (``ihconsv=1``) or the inverse barometer at the boundary (``inv_atm_bnd=1``), SCHISM uses these values.
* The sflux forcing period is padded by one day on each side once. It grew by another day on each side for every active sflux file, and lost its interval.
* The relative weights of ``rad`` and ``prc`` sflux sources are checked, not only ``air``.
* ``SfluxPrc`` has ``data_type`` ``sflux_prc`` (it was ``sflux_rad``).
* An ``SfluxAir`` source without a ``uri`` is an error; it silently used a test-data path.
* SCHISM now starts from the ``hotstart.nc`` written by ``boundary_conditions.hotstart_config``. ``opt.ihot`` stayed 0, so SCHISM cold-started and ignored the file: the check looked for a ``data.hotstart`` field that no longer exists. ``ihot`` is set to 1 when not set, and ``ihot=0`` with a hotstart is an error.
* About 40 ``param.nml`` parameters had no effect: they were written to ``&VERTICAL`` and ``&VEGETATION`` groups, but SCHISM only reads ``&CORE``, ``&OPT`` and ``&SCHOUT``. They include the backtracking limits (``s1_mxnbt``, ``s2_mxnbt``), ``rho0``, ``slr_rate``, ``iflux``, ``iharind`` and the vegetation model. They are now fields of ``opt`` and are written to ``&OPT``.
* ``veg_lai`` and ``veg_cw`` are integers, as SCHISM v5.13 and v5.14 declare them. Written as reals, now that SCHISM reads them, they stop the run with a namelist read error.
* ``schout.nhot=1`` no longer fails validation. The check of ``nhot_write`` read ``ihfskip`` and ``dt`` from ``schout`` instead of ``core``; it now follows SCHISM: ``nhot_write`` is a multiple of ``core.ihfskip`` with hotstart output, and of ``nspool_sta`` with station output.
* ``bctides.in`` has exactly one entry per open boundary of the mesh. It had one per key of ``boundaries`` up to the largest key, and a single ``5 5 0 0`` boundary when ``boundaries`` was empty. A missing or unknown open boundary is now an error that says which.
* ``bctides.in`` follows SCHISM's reader for every boundary type. Types read from files (elevation 1 and 4, discharge 1, tracers 1 and 4, relaxed velocity) no longer have comment lines inside the data; constant elevation and discharge (type 2) are one value, as SCHISM reads them, instead of none or one per node; Flather boundaries have a mean normal velocity per vertical level. The unread ``ncbn``/``nfluxf`` lines at the end are gone.
* Tidal constituents keep the order they are given in, so ``bctides.in`` is the same from one run to the next, and a single constituent works.
* ``TidalDataset.tide_interpolation_method`` is used; it was always bilinear.
* Boundary conditions without tidal data, and a ``TidalDataset`` without mean dynamic topography, no longer fail.
* Boundary files (``elev2D.th.nc``, ``uv3D.th.nc``, ``TEM_3D.th.nc``, ``SAL_3D.th.nc``) hold the nodes of the open boundaries that use them, as SCHISM reads them. They held all open boundary nodes, which SCHISM cannot read when only some boundaries use the file, for example an ocean boundary with a river. Each file is written once, from one source, set with ``SCHISMDataBoundary.open_boundaries``.

Deprecations
------------
* ``Param.vertical`` and ``Param.vegetation`` are replaced by ``Param.opt``. Configurations that still use them are accepted with a ``DeprecationWarning`` and their values are moved to ``opt``.

0.5.0 (2025-07-13)
___________________

New Features
------------
* Improved logging for SCHISM model components.
* Added string formatting methods for SCHISM components.
* Added backend testing and execution scripts.
* Added backend config examples and quickstart test script.
* Added backend demo notebook and documentation.
* Added support for multiple include_modules in docker.
* Added docker backend test and setup.

Bug Fixes
---------
* Fixed backend demo notebook.
* Fixed merge issues and improved PyLibs import handling.
* Fixed mounting workspace in docker.
* Fixed issues from merge and removed redundant code.

Internal Changes
----------------
* Refactored backend approach using strong typing.
* Improved backend docs and tutorial.
* Consolidated documentation and removed legacy sections.
* Improved INPUT file diagnostics in Docker container.


0.4.0 (2025-07-10)
___________________

New Features
------------
* Refactored SCHISM boundary conditions, unified SCHISMDataTides and SCHISMDataOcean into SCHISMDataBoundaryConditions.
* Added support for pyTMD for tidal forcing.
* Added tidal database and updated yaml of tidal runs.
* Added boundary condition examples and documentation.
* Added plotting utilities and improved grid plotting.
* Added support for v5.12 vegetation model.
* Added MDT specification for TidalDataset for Z0.

Bug Fixes
---------
* Fixed duplicated subfolder for oceanum-atlas in database.json.
* Fixed test cases for pyTMD compatibility.
* Fixed decorators and case handling for tidal API.
* Fixed missing station.in file for tidal examples.
* Fixed case test with new tidal API.

Internal Changes
----------------
* Restructured SCHISM boundary condition naming.
* Overwrote pre-refactor config files with post-refactor versions.
* Cleaned up code and tests.
* Updated enum types and documentation.



0.3.0 (2023-03-29)
___________________

Major refactor: redefinition of the entire codebase using pydantic models.
Separation of concerns between runtime information and model configuration.
Added model_type field and pydantic basegrid.
Added methods to store and dump original inputs to RompyBaseModel.
Added json support to CLI.
Added iso timedelta for JSON serialization.
Added DataPoint object and timeseries-based sources.
Added validator for hotfiles against timestep.

Bug Fixes
---------
Fixed issue with create_model in older versions of pydantic.
Fixed validator of IDLA to fix serialization issue.
Fixed path definition in new test.
Fixed schism serialization issues.
Fixed bug in string output of regular grid for SWAN.

Internal Changes
----------------
Removed convenience imports from core.
Promoted appdirs dependency from schism to main list.
Reordered imports.
Refactored intake source to prevent recursion with dask.
Cleaned up debug messages and removed redundant code.

0.1.0 (2023-MM-DD)
___________________

Initial release of rompy with basic functionality for coastal ocean model configuration and execution.
Provided example Jupyter notebooks for setup, evaluation, and visualization.
Basic support for SWAN and SCHISM models.


.. _`CSIRO`: https://www.csiro.au/en/
.. _`Oceanum`: https://oceanum.science/
