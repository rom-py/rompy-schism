import sys
from datetime import datetime
from pathlib import Path

from utils import compare_files

from rompy.core import DataBlob, TimeRange
from rompy.model import ModelRun
from rompy.schism import Inputs, SchismConfig

sys.path.append("../")


# from ..utils import compare_files

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
        config=SchismConfig(
            inputs=Inputs(
                hgrid_file=DataBlob(
                    id="hgrid_file", source=here / "test_data" / "hgrid_WWM.gr3"
                ),
                wwmbnd_file=DataBlob(
                    id="wwmbnd_file", source=here / "test_data" / "wwmbnd.gr3"
                ),
                filewave=DataBlob(
                    id="filewave",
                    source=here
                    / "test_data"
                    / "schism_bnd_spec_SWAN_500m_use_in_schism_2021Aug-Nov.nc",
                ),
            )
        ),
    )
    runtime.generate()
    for fname in ["param.nml", "wwminput.nml"]:
        compare_files(
            here / "reference_files" / runtime.run_id / fname,
            tmpdir / runtime.run_id / fname,
        )
