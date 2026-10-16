# Tutorial

The SCHISM tutorial takes you from zero to a tide and wind hindcast in seven lessons, on a model off Perth, Western Australia. The lessons are notebooks, rendered with their outputs on the [rompy notebooks site](https://rom-py.github.io/rompy-notebooks/schism-tutorial/), where you can also download them. Work through them in order.

```python exec="on"
from rompy_docs.notebooks import tutorial_table

print(tutorial_table("schism"))
```

The notebooks and their example data are in the [rompy-notebooks](https://github.com/rom-py/rompy-notebooks/tree/main/notebooks/schism) repository, so they run from a clone without further setup. The model runs need Docker, and are skipped without it.
