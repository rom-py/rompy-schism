from pathlib import Path

import pytest
from utils import compare_nmls

pytest.importorskip("rompy.schism")

from rompy.schism.namelists import ICE, ICM, MICE, PARAM, SEDIMENT

SAMPLE_DIR = (
    Path(__file__).parent
    / ".."
    / ".."
    / "rompy"
    / "schism"
    / "namelists"
    / "sample_inputs"
)


def test_namelists(tmp_path):
    for nml in [ICM, PARAM, SEDIMENT, MICE, ICE]:
        instance = nml()
        instance.write_nml(tmp_path)
        name = instance.__class__.__name__.lower()
        compare_nmls(tmp_path / f"{name}.nml", SAMPLE_DIR / f"{name}.nml")
