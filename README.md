---
title: "Rompy SCHISM Plugin"
---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15093426.svg)](https://doi.org/10.5281/zenodo.15093426)
[![GitHub Pages](https://github.com/rom-py/rompy-schism/actions/workflows/sphinx_docs_to_gh_pages.yaml/badge.svg)](https://rom-py.github.io/rompy-schism/)
[![PyPI version](https://img.shields.io/pypi/v/rompy-schism.svg)](https://pypi.org/project/rompy-schism/)
[![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/rom-py/rompy-schism/python-publish.yml)](https://github.com/rom-py/rompy-schism/actions)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rompy-schism)](https://pypistats.org/packages/rompy-schism)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rompy-schism)](https://pypi.org/project/rompy-schism/)

# Introduction

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

# Documentation

See https://rom-py.github.io/rompy-schism/

For information about the core rompy framework, see https://rom-py.github.io/rompy/

# Code Formatting and Pre-commit Hooks

This repository enforces Python code formatting using [black](https://github.com/psf/black) via the pre-commit framework.

To set up pre-commit hooks locally (required for all contributors)::

    pip install pre-commit
    pre-commit install

This will automatically check code formatting before each commit. To format your code manually, run::

    pre-commit run --all-files

All code must pass black formatting before it can be committed or merged.

# Versioning and Release

This project uses [tbump](https://github.com/dmerejkowsky/tbump) for version management.

To bump the version, run::

    tbump <new_version>

This will update the version in `src/rompy_schism/__init__.py`, commit the change, and optionally create a git tag.

tbump is included in the development requirements (`requirements_dev.txt`).

For more advanced configuration, see `tbump.toml` in the project root.

# Relevant packages

> - [rompy](https://github.com/rom-py/rompy)
> - [rompy-swan](https://github.com/rom-py/rompy-swan)
> - [rompy-schism](https://github.com/rom-py/rompy-schism)
> - [rompy-notebooks](https://github.com/rom-py/rompy-notebooks)