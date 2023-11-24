import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, field_validator, model_validator

from rompy.core import DataGrid
from rompy.core.time import TimeRange

logger = logging.getLogger(__name__)


class SCHISMDataAtmos(DataGrid):
    """This class is used to write SCHISM data from a dataset."""

    air_1_relative_weight: float = Field(
        1.0,
        description="air_[12]_relative_weight set the relative ratio between datasets '1' and '2'",
    )
    air_2_relative_weight: float = Field(
        99.0,
        description="air_[12]_relative_weight set the relative ratio between datasets '1' and '2'",
    )
    air_1_max_window_hours: float = Field(
        120.0,
        description="max. # of hours (offset from start time in each file) in each file of set '1'",
    )
    air_2_max_window_hours: float = Field(
        120.0,
        description="max. # of hours (offset from start time in each file) in each file of set '1'",
    )
    air_1_fail_if_missing: bool = Field(True, description="set '1' is mandatory")
    air_2_fail_if_missing: bool = Field(False, description="set '2' is optional")
    air_1_file: str = Field("sflux_air_1", description="file name for 1st set of 'air'")
    air_2_file: str = Field("sflux_air_2", description="file name for 2nd set of 'air'")
    uwind_name: str = Field("uwind", description="name of u-wind vel.")
    vwind_name: str = Field("vwind", description="name of v-wind vel.")
    prmsl_name: str = Field(
        "prmsl", description="name of air pressure (@MSL) variable in .nc file"
    )
    stmp_name: str = Field("stmp", description="name of surface air T")
    spfh_name: str = Field("spfh", description="name of specific humidity")
    rad_1_relative_weight: float = Field(1.0)
    rad_2_relative_weight: float = Field(99.0)
    rad_1_max_window_hours: float = Field(24.0)
    rad_2_max_window_hours: float = Field(24.0)
    rad_1_fail_if_missing: bool = Field(False)
    rad_2_fail_if_missing: bool = Field(False)
    rad_1_file: str = Field("sflux_rad_1")
    rad_2_file: str = Field("sflux_rad_2")
    dlwrf_name: str = Field("dlwrf")
    dswrf_name: str = Field("dswrf")
    prc_1_relative_weight: float = Field(1.0)
    prc_2_relative_weight: float = Field(99.0)
    prc_1_max_window_hours: float = Field(24.0)
    prc_2_max_window_hours: float = Field(24.0)
    prc_1_fail_if_missing: bool = Field(False)
    prc_2_fail_if_missing: bool = Field(False)
    prc_1_file: str = Field("sflux_prc_1")
    prc_2_file: str = Field("sflux_prc_2")
    prate_name: str = Field("prate", description="name of precipitation rate variable")

    def _set_variables(self):
        for variable in [
            "uwind",
            "vwind",
            "prmsl",
            "stmp",
            "spfh",
            "prate",
        ]:
            if not getattr(self, variable) is None:
                self.variables.append(variable)

    def __str__(self):
        return f"SCHISMAtmos"
