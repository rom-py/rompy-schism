# Mesh and vertical grid

[`SCHISMGrid`][rompy_schism.grid.SCHISMGrid] holds the mesh, `hgrid.gr3`, the vertical grid, `vgrid.in`, and the property files SCHISM reads with them, one value per node. When the workspace is generated, it copies the files you give and writes the others.

```python exec="on" session="mesh"
import tempfile

from rompy.logging import config as logging_config

logging_config.update(level="ERROR")
OUT_DIR = tempfile.mkdtemp()
```

## The mesh

`hgrid.gr3` lists the nodes (coordinates and depth, positive down), the elements (triangles, or quadrilaterals) and the boundary: which nodes form open boundaries and which form land. It is the same format as ADCIRC's `fort.14`. rompy-schism does not make meshes; make them with a meshing tool such as [gmsh](https://gmsh.info), [OceanMesh2D](https://github.com/CHLNDDEV/OceanMesh2D) or SMS ([Making a mesh](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/making_a_mesh/) shows one way).

Give the mesh as a [`DataBlob`][rompy.core.data.DataBlob]. `SCHISMGrid` reads it with [pylib](https://github.com/wzhengui/pylibs), available as `pylibs_hgrid`:

```python exec="on" source="above" result="text" session="mesh"
from rompy.core.data import DataBlob
from rompy_schism.grid import SCHISMGrid

grid = SCHISMGrid(hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"), drag=0.0025)
mesh = grid.pylibs_hgrid
print(f"{mesh.np} nodes, {mesh.ne} elements")
print(f"{grid.nob} open boundary, with {grid.nobn[0]} nodes")
```

The open boundaries are numbered in the order of `hgrid.gr3`, from 0. `bctides.in` has one entry per open boundary in that order, and the boundary files list the open boundary nodes in that order too; [Open boundaries and tides](boundaries.md) sets them by this number.

## Property files

Besides the mesh, SCHISM reads several files in the same format, with one value per node. Each field of `SCHISMGrid` takes a file, as a `DataBlob`, or a number, which is written to every node:

| Field | File | Default | Used for |
|---|---|---|---|
| `drag`, `rough` or `manning` | `drag.gr3`, `rough.gr3` or `manning.gr3` | None: one is required | Bottom friction: drag coefficient, roughness length (m) or Manning's n |
| `diffmin`, `diffmax` | `diffmin.gr3`, `diffmax.gr3` | 1e-6, 1.0 | Limits of the vertical diffusivity, for the turbulence closure in 3D |
| `albedo` | `albedo.gr3` | 0.15 | Heat exchange with the atmosphere |
| `watertype` | `watertype.gr3` | 1 | Light absorption in the water, for heat exchange |
| `windrot_geo2proj` | `windrot_geo2proj.gr3` | 0.0 | Rotation from geographic north to the mesh coordinates, for the wind |
| `hgridll`, `hgrid_WWM` | `hgrid.ll`, `hgrid_WWM.gr3` | Links to `hgrid.gr3` | The mesh in longitude and latitude, and for WWM |
| `wwmbnd` | `wwmbnd.gr3` | From the open boundaries | Open boundary flags for WWM |

The workspace also gets `tvd.prop`, which turns on the TVD transport scheme for every element.

```python exec="on" source="above" result="text" session="mesh"
from pathlib import Path

grid.get(Path(OUT_DIR) / "grid")
print(sorted(p.name for p in (Path(OUT_DIR) / "grid").iterdir()))
```

## Bottom friction

SCHISM always applies a bottom friction, so one of `drag`, `rough` and `manning` is required, and only one. The field you set also sets `opt.nchi`, which tells SCHISM which file to read: 0 for `drag.gr3`, -1 for `manning.gr3`, 1 for `rough.gr3`. A different `nchi` is an error.

A constant value works, but friction usually varies, for example with the bed or the depth, and rompy-schism logs a warning that a map is better. Write a property file with pylib's `write_hgrid`, with the values in place of the depths, and give it as a file:

```python exec="on" source="above" session="mesh"
import numpy as np

# Rougher in shallow water: 0.005 above 5 m depth, 0.0025 below 50 m
drag = np.interp(mesh.dp, [5.0, 50.0], [0.005, 0.0025])
drag_file = Path(OUT_DIR) / "drag.gr3"
mesh.write_hgrid(str(drag_file), value=drag)

grid = SCHISMGrid(hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"), drag=DataBlob(source=drag_file))
```

## The vertical grid

`vgrid.in` sets the vertical levels. Without a `vgrid`, the model is **2D**: depth-averaged, with a single layer (two levels).

```python exec="on" source="above" result="text" session="mesh"
print(f"3D: {grid.is_3d}, levels: {grid.nvrt}")
```

A **3D** model has more levels. SCHISM has two kinds of vertical grid:

- **SZ:** terrain-following (sigma) levels, optionally with fixed-depth (Z) levels below a transition depth. rompy-schism generates it with [`VgridGenerator`][rompy_schism.grid.VgridGenerator] (`vgrid_type="sz"`) or [`VGrid`][rompy_schism.vgrid.VGrid].
- **LSC²:** localised sigma coordinates, where the number of levels varies with the depth. It is made from the mesh depths with SCHISM's `gen_vqs` tool, not by rompy-schism. Give the resulting `vgrid.in` as a file: `vgrid=DataBlob(source="vgrid.in")`.

For SZ levels, `nvrt` is the number of levels, `h_c` the depth above which they are evenly spaced, and `theta_f` and `theta_b` concentrate them near the surface and, with `theta_b`, also near the bottom:

```python exec="on" source="above" result="text" session="mesh"
from rompy_schism.grid import VgridGenerator

grid_3d = SCHISMGrid(
    hgrid=DataBlob(source="tests/data/schism/hgrid.gr3"),
    vgrid=VgridGenerator(vgrid_type="sz", nvrt=20, h_c=10.0, theta_b=0.5, theta_f=3.0),
    drag=0.0025,
)
print(f"3D: {grid_3d.is_3d}, levels: {grid_3d.nvrt}")
```

A 3D model needs settings and data that a 2D model does not: see [Model settings](model-settings.md#barotropic-or-baroclinic) and, for temperature and salinity at the boundary and at the start, [Open boundaries](boundaries.md#from-an-ocean-model) and [Initial conditions](initial-conditions.md). [Baroclinic 3D model](https://rom-py.github.io/rompy-notebooks/notebooks/schism/examples/baroclinic_3d/) puts it together.

## Checking a mesh against the time step

SCHISM is semi-implicit, so its time step is not limited by the element size as in explicit models. The SCHISM manual instead recommends a time step long enough to keep numerical diffusion low: a Courant number (|u| + √(gh))·Δt/Δx above about 0.4. [Tutorial 2](https://rom-py.github.io/rompy-notebooks/notebooks/schism/tutorial/02_mesh_and_vertical_grid/) shows how to compute it for a mesh.
