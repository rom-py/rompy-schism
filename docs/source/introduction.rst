=================================
Introduction to rompy-schism
=================================

The rompy-schism package is a plugin for the rompy framework that provides support for configuring and running the SCHISM (Semi-implicit Cross-scale Hydroscience Integrated System Model) hydrodynamic model.

SCHISM is a community-supported modeling tool that simulates water circulation, waves, and associated transport and transformation processes across a range of scales from estuaries to the global ocean. It uses unstructured grids in the horizontal plane and terrain-following coordinates in the vertical, allowing for efficient and accurate modeling of complex coastlines and bathymetry.

Rompy-SCHISM Integration
------------------------

The rompy-schism package provides a Pythonic interface to configure SCHISM model runs using the rompy framework. It includes:

- Pydantic models for all SCHISM namelists and configuration parameters
- Automatic generation of SCHISM input files from Python objects
- Integration with rompy's data handling and grid systems
- Support for various SCHISM components including:
  - Grid configuration
  - Sflux meteorological forcing
  - Boundary conditions (tidal, ocean, river, nested)
  - Hotstart file generation
  - Physics and numerical parameters
  - Output specifications

The package is organized into modules that correspond to SCHISM's namelist structure:

- **param**: Core model parameters
- **ice**: Ice model parameters
- **mice**: Multi-category ice model parameters
- **icm**: Integrated Compartment Model parameters
- **sediment**: Sediment transport parameters
- **cosine**: Cosine model parameters
- **wwminput**: WWM (Wave Watch III) input parameters

For more information on the core rompy concepts that this package builds upon, see the :external:doc:`rompy documentation <index>`.