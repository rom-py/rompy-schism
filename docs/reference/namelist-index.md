# Namelist index

Every parameter rompy-schism writes to `param.nml` and `wwminput.nml`, with the field that sets it, by namelist group. The index is generated from the code when the docs are built. Array entries such as `iof_hydro(1)` are fields named `iof_hydro__1`. The [SCHISM manual](https://schism-dev.github.io/schism/master/input-output/param.html) describes each parameter in full.

The namelists of the other modules (sediment, ice, ICM, CoSiNE) are listed on the [Modules](modules.md) page.

```python exec="on"
import sys

sys.path.insert(0, "scripts")
from namelist_index import namelist_index

print(namelist_index())
```
