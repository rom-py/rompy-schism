import sys
from datetime import datetime
from pathlib import Path
from shutil import rmtree

import pytest

pytest.importorskip("rompy.schism")
from utils import compare_files

from rompy.core import DataBlob, TimeRange
from rompy.model import ModelRun
from rompy.schism import SCHISMConfig, SCHISMGrid

here = Path(__file__).parent


def test_schism_render(tmpdir):
    """Test the swantemplate function."""
    run_id = "test_schism"
    period = TimeRange(
        start=datetime(2021, 8, 1, 0), end=datetime(2021, 11, 29, 0), interval="15M"
    )
    runtime = ModelRun(
        period=period,
        run_id=run_id,
        output_dir=str(tmpdir),
        config=SCHISMConfig(
            grid=SCHISMGrid(
                hgrid=DataBlob(id="hgrid", source=here / "test_data" / "hgrid.gr3"),
                drag=1,
            )
        ),
    )
    runtime.generate()

    # for fname in ["param.nml", "wwminput.nml"]:
    #     compare_files(
    #         here / "reference_files" / runtime.run_id / fname,
    #         tmpdir / runtime.run_id / fname,
    #     )
    #     # assert file exists
    #     for fname in [
    #         "diffmax.gr3",
    #         "diffmin.gr3",
    #         "hgrid.gr3",
    #         "hgrid_WWM.gr3",
    #         # "drag.gr3",
    #         "manning.gr3",
    #         "schism_bnd_spec_SWAN_500m_use_in_schism_2021Aug-Nov.nc",
    #         "vgrid.in",
    #         "wwmbnd.gr3",
    #     ]:
    #         assert (tmpdir / runtime.run_id / fname).exists()


if __name__ == "__main__":
    tmpdir = "test_schism"
    rmtree(tmpdir, ignore_errors=True)
    test_schism_render(tmpdir)
