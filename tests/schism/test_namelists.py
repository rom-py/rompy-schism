from pathlib import Path

import pytest

from rompy.schism.namelists import ICE, ICM, MICE, PARAM, SEDIMENT
from rompy.schism.namelists.generate_models import nml_to_dict

SAMPLE_DIR = (
    Path(__file__).parent
    / ".."
    / ".."
    / "rompy"
    / "schism"
    / "namelists"
    / "sample_inputs"
)


# funcition to step through the namelist and compare the values
# ipdb if they are not equal
def compare_nmls_values(nml1, nml2):
    for key, value in nml1.items():
        if isinstance(value, dict):
            compare_nmls_values(value, nml2[key])
        if value != nml2[key]:
            import ipdb

            print(key, value, nml2[key])
            ipdb.set_trace()


def compare_nmls(nml1, nml2):
    d1 = nml_to_dict(nml1)
    d2 = nml_to_dict(nml2)
    d1.pop("description")
    d2.pop("description")
    compare_nmls_values(d1, d2)
    assert d1 == d2


def test_namelists(tmp_path):
    for nml in [ICM, PARAM, SEDIMENT, MICE, ICE]:
        instance = nml()
        instance.write_nml(tmp_path)
        name = instance.__class__.__name__.lower()
        compare_nmls(tmp_path / f"{name}.nml", SAMPLE_DIR / f"{name}.nml")
    # ice = ICE()
    # ice.write_nml(tmp_path)
    # compare_nmls(tmp_path / "ice.nml", SAMPLE_DIR / "ice.nml")
