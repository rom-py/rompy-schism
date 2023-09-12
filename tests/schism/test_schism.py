from datetime import datetime
from pathlib import Path

from rompy.core import TimeRange
from rompy.model import ModelRun
from rompy.schism import SchismConfig
from rompy.schism.config import DEFAULT_TEMPLATE

# from ..utils import compare_files

here = Path(__file__).parent


def test_schism_render(tmpdir):
    """Test the swantemplate function."""
    time = TimeRange(
        start=datetime(2020, 2, 21, 4), end=datetime(2020, 2, 24, 4), interval="15M"
    )
    runtime = ModelRun(
        run_id="test_schism",
        output_dir=str(tmpdir),
        config=SchismConfig(),
    )
    runtime.generate()
    # compare_files(
    #     here / "simulations" / "test_swan_ref" / "INPUT_NEW",
    #     tmpdir / runtime.run_id / "INPUT",
    # )


# def test_schism_config(tmpdir):
#     """Test the swantemplate function."""
#     config = SchismConfig()
