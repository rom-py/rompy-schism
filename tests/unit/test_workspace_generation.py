"""Workspace files written by the grid and namelists."""

from rompy.core.time import TimeRange

from rompy_schism.namelists import NML, Param
from rompy_schism.namelists.wwminput import Wwminput

PERIOD = TimeRange(start="2023-01-01T00", end="2023-01-02T00", interval="1h")


def test_wwminput_is_written_only_when_set(tmp_path):
    nml = NML(param=Param())
    nml.update_times(PERIOD)
    nml.write_nml(tmp_path)
    assert (tmp_path / "param.nml").exists()
    assert not (tmp_path / "wwminput.nml").exists()


def test_wwminput_gets_the_run_times(tmp_path):
    nml = NML(param=Param(), wwminput=Wwminput())
    nml.update_times(PERIOD)
    assert nml.wwminput.proc.begtc == "20230101.000000"
    assert nml.wwminput.proc.endtc == "20230102.000000"


def test_grid_files_can_be_generated_twice(grid2d, tmp_path):
    grid2d.get(tmp_path)
    grid2d.get(tmp_path)
    assert (tmp_path / "hgrid.ll").is_symlink()
    assert (tmp_path / "hgrid_WWM.gr3").is_symlink()
