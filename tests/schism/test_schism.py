import sys
from datetime import datetime
from pathlib import Path

from utils import compare_files

from rompy.core import TimeRange
from rompy.model import ModelRun
from rompy.schism import SchismConfig
from rompy.schism.config import DEFAULT_TEMPLATE

sys.path.append("../")


# from ..utils import compare_files

here = Path(__file__).parent


def test_schism_render(tmpdir):
    """Test the swantemplate function."""
    run_id = "test_schism"
    time = TimeRange(
        start=datetime(2020, 2, 21, 4), end=datetime(2020, 2, 24, 4), interval="15M"
    )
    runtime = ModelRun(
        run_id=run_id,
        output_dir=str(tmpdir),
        config=SchismConfig(),
    )
    runtime.generate()
    for fname in ["param.nml", "wwminput.nml"]:
        compare_files(
            here / "reference_files" / runtime.run_id / fname,
            tmpdir / runtime.run_id / fname,
        )
