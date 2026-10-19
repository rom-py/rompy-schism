"""SCHISM reads the friction file that the grid writes."""

import pytest
from pydantic import ValidationError
from rompy.core.data import DataBlob

from rompy_schism.config import SCHISMConfig
from rompy_schism.grid import SCHISMGrid
from rompy_schism.namelists import NML, Param


def config(hgrid_path, friction: dict, opt: dict | None = None) -> SCHISMConfig:
    grid = SCHISMGrid(hgrid=DataBlob(source=hgrid_path), **friction)
    return SCHISMConfig(grid=grid, nml=NML(param=Param(opt=opt or {})))


@pytest.mark.parametrize(
    "friction, nchi",
    [({"drag": 0.0025}, 0), ({"manning": 0.025}, -1), ({"rough": 0.001}, 1)],
    ids=["drag", "manning", "rough"],
)
def test_nchi_follows_the_grid_friction(hgrid_path, friction, nchi):
    assert config(hgrid_path, friction).nml.param.opt.nchi == nchi


def test_matching_nchi_is_kept(hgrid_path):
    assert config(hgrid_path, {"manning": 0.025}, {"nchi": -1}).nml.param.opt.nchi == -1


def test_nchi_for_another_friction_file_raises(hgrid_path):
    with pytest.raises(ValidationError, match="set nchi to -1"):
        config(hgrid_path, {"manning": 0.025}, {"nchi": 0})
