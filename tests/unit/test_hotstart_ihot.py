"""SCHISM starts from the hotstart.nc that the boundary conditions write."""

import pytest
from pydantic import ValidationError

from rompy_schism.config import SCHISMConfig
from rompy_schism.data import HotstartConfig, SCHISMData, SCHISMDataBoundaryConditions
from rompy_schism.namelists import NML, Param


def config(grid2d, hotstart: bool, opt: dict | None = None) -> SCHISMConfig:
    conditions = SCHISMDataBoundaryConditions(
        hotstart_config=HotstartConfig(enabled=hotstart)
    )
    return SCHISMConfig(
        grid=grid2d,
        data=SCHISMData(boundary_conditions=conditions),
        nml=NML(param=Param(opt=opt or {})),
    )


def test_hotstart_sets_ihot_when_unset(grid2d):
    assert config(grid2d, hotstart=True).nml.param.opt.ihot == 1


def test_hotstart_keeps_ihot_set_by_the_user(grid2d):
    assert config(grid2d, hotstart=True, opt={"ihot": 2}).nml.param.opt.ihot == 2


def test_hotstart_with_ihot_zero_raises(grid2d):
    with pytest.raises(ValidationError, match="ihot=0 makes SCHISM ignore"):
        config(grid2d, hotstart=True, opt={"ihot": 0})


def test_hotstart_creates_missing_namelist_groups(grid2d):
    conditions = SCHISMDataBoundaryConditions(
        hotstart_config=HotstartConfig(enabled=True)
    )
    cfg = SCHISMConfig(
        grid=grid2d,
        data=SCHISMData(boundary_conditions=conditions),
        nml=NML(),
    )
    assert cfg.nml.param.opt.ihot == 1


def test_hotstart_creates_missing_opt_group(grid2d):
    conditions = SCHISMDataBoundaryConditions(
        hotstart_config=HotstartConfig(enabled=True)
    )
    cfg = SCHISMConfig(
        grid=grid2d,
        data=SCHISMData(boundary_conditions=conditions),
        nml=NML(param=Param(opt=None)),
    )
    assert cfg.nml.param.opt.ihot == 1


def test_hotstart_creates_missing_namelist_when_disabled(grid2d):
    conditions = SCHISMDataBoundaryConditions(
        hotstart_config=HotstartConfig(enabled=True)
    )
    cfg = SCHISMConfig(
        grid=grid2d,
        data=SCHISMData(boundary_conditions=conditions),
        nml=None,
    )
    assert cfg.nml.param.opt.ihot == 1


def test_no_hotstart_leaves_ihot(grid2d):
    assert config(grid2d, hotstart=False).nml.param.opt.ihot == 0
