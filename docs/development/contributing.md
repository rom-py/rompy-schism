# Contributing

Contributions are welcome: bug reports and ideas as [GitHub issues](https://github.com/rom-py/rompy-schism/issues), and changes as pull requests.

## Development setup

```bash
git clone https://github.com/rom-py/rompy-schism.git
cd rompy-schism
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Tests and style

```bash
pytest                    # the test suite; downloads the test data on first run
black --check src tests   # formatting, also run by the pre-commit hook
```

The test data (`tests/data`) is downloaded from [rompy-test-data](https://github.com/rom-py/rompy-test-data) the first time the tests run. Tests that run SCHISM need Docker, and are skipped without it.

Test what each part writes (the namelists, `bctides.in`, the boundary and grid files), and each validator in both directions (accepted and rejected).

## How the code is organised

| Module | Holds |
|---|---|
| `config.py` | `SCHISMConfig`, and the checks across its parts |
| `grid.py`, `vgrid.py` | The mesh, the property files and the vertical grid |
| `data.py` | The data interfaces: boundary conditions, sflux, waves, hotstart |
| `boundary_core.py`, `bctides.py` | The boundary types, the tidal dataset, and writing `bctides.in` |
| `namelists/` | One module per namelist: `param.py`, `wwminput.py`, and those of the other modules |
| `schema.py` | The SCHISM versions `schema_version` accepts |

When changing them:

- **Each part writes exactly its files.** The configuration does not rewrite what a part writes. Settings that depend on each other are checked by a validator on `SCHISMConfig`, which raises an error that says what to use instead.
- **Fill in only what is unset.** Parameters that follow from the rest of the configuration (`nchi` from the grid's friction, `ihot` from the hotstart) are set when the user leaves them unset; a value that contradicts the rest is an error.
- **Namelist fields use SCHISM's names and defaults**, with a description from SCHISM's sample namelist. Array entries are fields named `name__i`, written `name(i)`.

## SCHISM versions

`schema_version` on `SCHISMConfig` selects the parameters of one SCHISM release, for the whole configuration. Do not version nested namelists separately. A new version must:

1. add a `schema_version` value in `schema.py`;
2. leave the default for configurations without a version at `schism-v5.13`, so existing configurations keep their meaning;
3. define what changes: defaults, validation and what is written;
4. migrate any parameter whose meaning changes;
5. add YAML round-trip tests;
6. test the generated files against a matching SCHISM release, for example with a [Docker image](../user-guide/running.md#with-docker) of it.

## Documentation

The docs are built with [ProperDocs](https://github.com/ProperDocs/properdocs) (a maintained fork of MkDocs 1.6), Material and mkdocstrings, using the shared configuration from [rompy-docs](https://github.com/rom-py/rompy-docs):

```bash
pip install -e ".[docs,test]"
python -c "import tests.conftest"           # download the example data into tests/data
tar -xzf tests/data/schism/tides/oceanum-atlas.tar.gz -C tests/data/schism/tides
properdocs serve -f mkdocs.yml              # live preview
properdocs build --strict -f mkdocs.yml     # what CI runs
rompy-docs check -f mkdocs.yml              # compare with the shared configuration
```

- **The reference is generated** from the docstrings and field descriptions, and lists the fields each class inherits from rompy. The [namelist index](../reference/namelist-index.md) is generated from the namelist models.
- **Examples in pages run when the docs are built**, as markdown-exec blocks (`` ```python exec="on" source="above" result="text" ``), against the test data. A failing example fails the strict build. Use a plain `python` block only for examples that need SCHISM, Docker or data not in the test data.
- **Link classes** with `` [`SCHISMGrid`][rompy_schism.grid.SCHISMGrid] ``. rompy classes resolve to the rompy site.
- **Notebooks** live in [rompy-notebooks](https://github.com/rom-py/rompy-notebooks). The Tutorial and Examples pages list them from the inventory that site publishes.
- To build against an unpublished rompy site, set `ROMPY_INVENTORY` to its `objects.inv` (a URL, or `file://` path).

## Pull requests

1. Create a branch from `main`.
2. Make the change, with tests and docs.
3. Run the tests and the strict docs build.
4. Add the change to the Unreleased section of [`HISTORY.rst`](https://github.com/rom-py/rompy-schism/blob/main/HISTORY.rst), under Breaking Changes, New Features, Bug Fixes or Deprecations.
5. Open a pull request.

## Releases

The version is managed with [tbump](https://github.com/dmerejkowsky/tbump): `tbump <new_version>` updates `src/rompy_schism/__init__.py`, commits and tags `v<new_version>`. Move the Unreleased section of `HISTORY.rst` to the new version first. Publishing a GitHub release for the tag publishes the package on PyPI.

The SCHISM Docker image is built from `docker/Dockerfile` for each release listed in `docker/releases.txt`; see `docker/README.md`.
