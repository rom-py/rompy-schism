# Running SCHISM

rompy-schism writes the workspace; SCHISM runs it. This page covers how SCHISM runs, the Docker image, your own build, and how to tell whether a run succeeded.

## How SCHISM runs

SCHISM always runs under MPI, from inside the workspace, which must have an `outputs/` folder (the template creates it):

```bash
mpirun -n 6 schism 2
```

The argument is the number of **scribes**, processes that only write output; the others compute. `mpirun -n 6 schism 2` runs 4 compute processes and 2 scribes. SCHISM needs one scribe for all its 2D outputs, one for the vertical coordinates, and one per 3D output variable, two for vectors: see [Scribes](output.md#scribes).

## With Docker

The public image `ghcr.io/rom-py/schism` has SCHISM v5.13.0 with MPI and NetCDF:

| Command | What runs |
|---|---|
| `schism` | Hydrodynamics |
| `schism_wwm` | Hydrodynamics coupled with the WWM wave model ([Waves](waves.md)) |
| `combine_hotstart7` | Merges the hotstart files of all processes ([Chained runs](initial-conditions.md#chained-runs)) |

rompy's [`DockerConfig`][rompy.backends.config.DockerConfig] mounts the workspace in the container and runs `mpiexec -n cpu executable` in it:

```python
from rompy.backends import DockerConfig

backend = DockerConfig(
    image="ghcr.io/rom-py/schism:5.13.0", executable="schism 2", mpiexec="mpirun", cpu=6
)
modelrun.run(backend, workspace_dir=workspace)
```

The same from a terminal, in the workspace folder:

```bash
docker run --rm -v "$PWD":/data -w /data ghcr.io/rom-py/schism:5.13.0 mpirun -n 6 schism 2
```

More processes than the machine has cores are fine: the image allows it, since the scribes mostly wait. Pin the version tag for reproducible runs, and use the matching [`schema_version`](configuration.md#schism-versions).

## With your own SCHISM

Compile SCHISM from [its source](https://github.com/schism-dev/schism) with the modules you need ([compilation guide](https://schism-dev.github.io/schism/master/getting-started/compilation.html)). rompy's [`LocalConfig`][rompy.backends.config.LocalConfig] runs a command in the workspace. The executable's name depends on how it was built:

```python
from rompy.backends import LocalConfig

modelrun.run(LocalConfig(command="mpirun -n 6 pschism_TVD-VL 2"), workspace_dir=workspace)
```

On a cluster, run the same command from your job script, in the generated workspace.

## From the command line

`rompy run` generates and runs a YAML configuration with a backend from a second YAML file:

```yaml
type: docker
image: ghcr.io/rom-py/schism:5.13.0
executable: schism 2
mpiexec: mpirun
cpu: 6
```

```bash
rompy run model.yml --backend-config docker.yml
```

## Checking a run

!!! warning "SCHISM's exit status"
    SCHISM exits with success even when it stops on an error, so `modelrun.run()`, which relies on the exit status, returns `True` either way.

Check SCHISM's own files instead:

- `outputs/mirror.out`, the log, ends with "Run completed successfully";
- `outputs/fatal.error` holds the reason SCHISM stopped;
- `outputs/nonfatal_*` hold warnings.

```python
from pathlib import Path


def run_completed(workspace: Path) -> bool:
    """Check SCHISM's log: SCHISM exits with success even when it stops on an error."""
    mirror = workspace / "outputs" / "mirror.out"
    if mirror.exists() and "Run completed successfully" in mirror.read_text():
        return True
    fatal = workspace / "outputs" / "fatal.error"
    print(fatal.read_text() if fatal.exists() else "SCHISM did not finish")
    return False
```

Besides the results, `outputs/` has files about the run:

| File | Contents |
|---|---|
| `mirror.out` | The log: setup, each time step, warnings, the final message |
| `param.out.nml` | The settings SCHISM used, including its own defaults |
| `fatal.error`, `nonfatal_*` | Errors and warnings |
| `maxelev_*`, `maxdahv_*` | Maximum water level and depth-averaged velocity at each node, per process |
| `total.out`, `total_TR.out` | Volume and tracer mass over time |

[Running SCHISM](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/running_schism/) runs a model with each backend, and shows a failed run.
