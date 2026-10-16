# rompy-schism

[![PyPI version](https://img.shields.io/pypi/v/rompy-schism.svg)](https://pypi.org/project/rompy-schism/)
[![Tests](https://github.com/rom-py/rompy-schism/actions/workflows/testing.yml/badge.svg)](https://github.com/rom-py/rompy-schism/actions/workflows/testing.yml)
[![Documentation](https://github.com/rom-py/rompy-schism/actions/workflows/build-docs.yml/badge.svg)](https://rom-py.github.io/rompy-schism/)
[![Python versions](https://img.shields.io/pypi/pyversions/rompy-schism)](https://pypi.org/project/rompy-schism/)

rompy-schism sets up and runs the [SCHISM](https://schism-dev.github.io/schism/master/) ocean model from Python or YAML. It is the SCHISM plugin of [rompy](https://rom-py.github.io/rompy/), which provides the run period, data sources, model runs and backends shared by all models.

SCHISM computes water levels, currents and, in 3D, temperature and salinity on an unstructured mesh. rompy-schism describes its input files as validated Python objects and writes them:

- **The mesh and the files that go with it**: `hgrid.gr3`, the vertical grid, bottom friction and the other property files.
- **Open boundaries from tides and ocean models**: tidal constituents from a tidal database, and boundary files interpolated from ocean model output.
- **Forcing and initial conditions from data**: atmospheric forcing (sflux), wave spectra for WWM, and `hotstart.nc` from an ocean model.
- **Model settings with SCHISM's names**: `param.nml` and the module namelists, for a chosen SCHISM version.

## Documentation

- Documentation: https://rom-py.github.io/rompy-schism/
- Tutorial and examples (notebooks): https://rom-py.github.io/rompy-notebooks/schism-tutorial/
- Changelog: [HISTORY.rst](https://github.com/rom-py/rompy-schism/blob/main/HISTORY.rst)

## Installation

```bash
pip install rompy-schism
```

SCHISM itself is not included. The public Docker image `ghcr.io/rom-py/schism` runs the generated workspaces; see [Running SCHISM](https://rom-py.github.io/rompy-schism/user-guide/running/).

## Contributing

See the [contributing guide](https://rom-py.github.io/rompy-schism/development/contributing/).

## License

Apache Software License 2.0. See [LICENSE](https://github.com/rom-py/rompy-schism/blob/main/LICENSE).
