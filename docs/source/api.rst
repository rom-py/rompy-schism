===============================
ROMPY-SCHISM API Documentation
===============================

Grids
------

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.grid.SCHISMGrid

Data
-----

Supporting objects for SCHISM data files

.. autosummary::
   :nosignatures:
   :toctree: _generated/
   rompy_schism.data.SfluxSource
   rompy_schism.data.SfluxAir
   rompy_schism.data.SfluxRad
   rompy_schism.data.SfluxPrc
   rompy_schism.data.SCHISMDataBoundary

Main objects

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.data.SCHISMDataSflux
   rompy_schism.data.SCHISMDataWave
   rompy_schism.data.SCHISMDataBoundaryConditions
   rompy_schism.data.HotstartConfig
   rompy_schism.data.SCHISMData

Boundary Conditions
-------------------

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.data.SCHISMDataBoundaryConditions
   rompy_schism.data.BoundarySetupWithSource
   rompy_schism.boundary_conditions.create_tidal_only_boundary_config
   rompy_schism.boundary_conditions.create_hybrid_boundary_config
   rompy_schism.boundary_conditions.create_river_boundary_config
   rompy_schism.boundary_conditions.create_nested_boundary_config


Config Minimal (no longer maintained)
--------------------------------------

This object has been implemented to meet the initial operational requirements of CSIRO. It is likely that this will be superceded by the full implementation.

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.config.SchismCSIROConfig

Full Namelist Implementation
-----------------------------

This object implements a set of models for each namelist and assembles a config object using this group of models.  This is curently only partly implemented.

PARAM
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.param.Core
   rompy_schism.namelists.param.Opt
   rompy_schism.namelists.param.Schout
   rompy_schism.namelists.param.Vertical
   rompy_schism.namelists.param.Param

ICE
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.ice.Ice_in
   rompy_schism.namelists.ice.Ice

MICE
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.mice.Mice_in
   rompy_schism.namelists.mice.Mice

ICM
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.icm.Bag
   rompy_schism.namelists.icm.Core
   rompy_schism.namelists.icm.Ero
   rompy_schism.namelists.icm.Marco
   rompy_schism.namelists.icm.Ph_icm
   rompy_schism.namelists.icm.Poc
   rompy_schism.namelists.icm.Sav
   rompy_schism.namelists.icm.Sfm
   rompy_schism.namelists.icm.Silica
   rompy_schism.namelists.icm.Stem
   rompy_schism.namelists.icm.Veg
   rompy_schism.namelists.icm.Zb
   rompy_schism.namelists.icm.Icm

SEDIMENT
~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.sediment.Sed_opt
   rompy_schism.namelists.sediment.Sed_core
   rompy_schism.namelists.sediment.Sediment


COSINE
~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.cosine.Core
   rompy_schism.namelists.cosine.Marco
   rompy_schism.namelists.cosine.Misc
   rompy_schism.namelists.cosine.Cosine


WWMINPUT
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.wwminput.Coupl
   rompy_schism.namelists.wwminput.Engs
   rompy_schism.namelists.wwminput.Grid
   rompy_schism.namelists.wwminput.History
   rompy_schism.namelists.wwminput.Hotfile
   rompy_schism.namelists.wwminput.Init
   rompy_schism.namelists.wwminput.Nesting
   rompy_schism.namelists.wwminput.Nums
   rompy_schism.namelists.wwminput.Petscoptions
   rompy_schism.namelists.wwminput.Proc
   rompy_schism.namelists.wwminput.Station
   rompy_schism.namelists.wwminput.Wwminput


NML
~~~~~

This is the full namelist object that is the container for all the other namelist objects.

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.namelists.NML




Config Object
~~~~~~~~~~~~~~


.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy_schism.config.SCHISMConfig
