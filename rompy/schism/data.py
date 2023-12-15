import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, field_validator, model_validator

from rompy.core import DataGrid, RompyBaseModel
from rompy.core.boundary import (
    BoundaryWaveStation,
    DataBoundary,
    SourceFile,
    SourceWavespectra,
)
from rompy.core.time import TimeRange
from rompy.schism.grid import SCHISMGrid2D, SCHISMGrid3D
from rompy.utils import total_seconds

from .namelists import Sflux_Inputs

logger = logging.getLogger(__name__)


GRID_TYPES = Union[SCHISMGrid2D, SCHISMGrid3D]


class SfluxSource(DataGrid):
    """This is a single variable source for and sflux input"""

    id: str = Field(..., description="id of the source", choices=["air", "rad", "prc"])
    relative_weight: float = Field(
        1.0,
        description="relative weight of the source file if two files are provided",
    )
    max_window_hours: float = Field(
        120.0,
        description="maximum number of hours (offset from start time in each file) in each file of set 1",
    )
    fail_if_missing: bool = Field(
        True, description="Fail if the source file is missing"
    )

    @property
    def outfile(self) -> str:
        # TODO - filenumber is. Hardcoded to 1 for now.
        return f'sflux_{self.id}.{str(1).rjust(4, "0")}.nc'

    @property
    def namelist(self) -> dict:
        ret = {}
        for key, value in self.model_dump().items():
            if key in ["relative_weight", "max_window_hours", "fail_if_missing"]:
                ret.update({f"{self.id}_{key}": value})
        ret.update({f"{self.id}_file": self.outfile})
        return ret


class SfluxAir(SfluxSource):
    """This is a single variable source for and sflux input"""

    uwind_name: SfluxSource = Field(
        None,
        description="name of zonal wind variable in source",
    )
    vwind_name: SfluxSource = Field(
        None,
        description="name of meridional wind variable in source",
    )
    prmsl_name: SfluxSource = Field(
        None,
        description="name of mean sea level pressure variable in source",
    )
    stmp_name: SfluxSource = Field(
        None,
        description="name of surface air temperature variable in source",
    )
    spfh_name: SfluxSource = Field(
        None,
        description="name of specific humidity variable in source",
    )

    def _set_variables(self) -> None:
        for variable in [
            "uwind_name",
            "vwind_name",
            "prmsl_name",
            "stmp_name",
            "spfh_name",
        ]:
            if getattr(self, variable) is not None:
                self.variables.append(getattr(self, variable))


class SfluxRad(SfluxSource):
    """This is a single variable source for and sflux input"""

    dlwrf_name: SfluxSource = Field(
        None,
        description="name of downward long wave radiation variable in source",
    )
    dswrf_name: SfluxSource = Field(
        None,
        description="name of downward short wave radiation variable in source",
    )

    def _set_variables(self) -> None:
        for variable in ["dlwrf_name", "dswrf_name"]:
            if getattr(self, variable) is not None:
                self.variables.append(getattr(self, variable))


class SfluxPrc(SfluxSource):
    """This is a single variable source for and sflux input"""

    prate_name: SfluxSource = Field(
        None,
        description="name of precipitation rate variable in source",
    )

    def _set_variables(self) -> None:
        self.variables = [self.prate_name]


class SCHISMDataSflux(RompyBaseModel):
    """This class is used to write SCHISM sflux data from a dataset.

    Arguments:
        air_1 (SfluxSource): The sflux air source 1.
        air_2 (SfluxSource): The sflux air source 2.
        rad_1 (SfluxSource): The sflux rad source 1.
        rad_2 (SfluxSource): The sflux rad source 2.
        prc_1 (SfluxSource): The sflux prc source 1.
        prc_2 (SfluxSource): The sflux prc source 2.

    """

    air_1: SfluxSource = Field(None, description="sflux air source 1")
    air_2: SfluxSource = Field(None, description="sflux air source 2")
    rad_1: SfluxSource = Field(None, description="sflux rad source 1")
    rad_2: SfluxSource = Field(None, description="sflux rad source 2")
    prc_1: SfluxSource = Field(None, description="sflux prc source 1")
    prc_2: SfluxSource = Field(None, description="sflux prc source 2")

    def get(
        self,
        destdir: str | Path,
        grid: Optional[GRID_TYPES] = None,
        time: Optional[TimeRange] = None,
    ) -> Path:
        """Writes SCHISM sflux data from a dataset.

        Args:
            destdir (str | Path): The destination directory to write the sflux data.
            grid (Optional[GRID_TYPES], optional): The grid type. Defaults to None.
            time (Optional[TimeRange], optional): The time range. Defaults to None.

        Returns:
            Path: The path to the written sflux data.

        """
        namelistargs = {}
        for variable in ["air_1", "air_2", "rad_1", "rad_2", "prc_1", "prc_2"]:
            data = getattr(self, variable)
            if data is None:
                continue
            namelistargs.update(data.namelist)
            data.get(destdir, grid, time)
        Sflux_Inputs(**namelistargs).write_nml(destdir / "sflux")

    @model_validator(mode="after")
    def check_weights(cls, v):
        """Check that relative weights for each pair add to 1.

        Args:
            cls: The class.
            v: The variable.

        Raises:
            ValueError: If the relative weights for any variable do not add up to 1.0.

        """
        for variable in ["air", "rad", "prc"]:
            weight = 0
            active = False
            for i in [1, 2]:
                data = getattr(v, f"{variable}_{i}")
                if data is None:
                    continue
                if data.fail_if_missing:
                    continue
                weight += data.relative_weight
                active = True
            if active and weight != 1.0:
                raise ValueError(
                    f"Relative weights for {variable} do not add to 1.0: {weight}"
                )


class SCHISMDataWave(BoundaryWaveStation):
    """This class is used to write SCHISM data from a dataset."""

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid2D | SCHISMGrid3D,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write the selected boundary data to a netcdf file.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid2D | SCHISMGrid3D
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        if self.crop_data and time is not None:
            self._filter_time(time)
        ds = self._sel_boundary(grid)
        outfile = Path(destdir) / f"{self.id}.nc"
        ds.spec.to_ww3(outfile)
        return outfile

    def __str__(self):
        return f"SCHISMDataWave"


class SCHISMDataBoundary(DataBoundary):
    """This class is used to extract ocean boundary data  griddd dataset at all open
    boundary nodes."""

    id: str = Field(
        "bnd",
        description="SCHISM th id of the source",
        choices=["elev2D", "uv3D", "TEM_3D", "SAL_3D", "bnd"],
    )
    variable: str = Field(..., description="variable name in the dataset")

    @model_validator(mode="after")
    def _set_variables(cls, v):
        v.variables = [v.variable]
        return v

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid2D | SCHISMGrid3D,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write the selected boundary data to a netcdf file.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid2D | SCHISMGrid3D
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        # prepare xarray.Dataset and save forcing netCDF file
        if self.crop_data and time is not None:
            self._filter_time(time)
        ds = self._sel_boundary(grid)
        dt = total_seconds((ds.time[1] - ds.time[0]).values)
        times = np.arange(0, ds.time.size) * dt
        time_series = np.expand_dims(ds[self.variable].values, axis=(2, 3))

        schism_ds = xr.Dataset(
            coords={
                "time": times,
                "nOpenBndNodes": np.arange(0, ds.xlon.size),
                "nComponents": np.array([1]),
                "one": np.array([1]),
            },
            data_vars={
                "time_step": (("one"), np.array([dt])),
                "time_series": (
                    ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                    time_series,
                ),
            },
        )
        schism_ds["time_step"] = schism_ds.time_step.assign_attrs(
            {"long_name": "time_step"}
        )
        schism_ds["time"] = schism_ds.time.assign_attrs(
            {"long_name": "simulation_time"}
        )
        schism_ds["time_series"] = schism_ds.time_series.assign_attrs(
            {"long_name": ds[self.variable].attrs["long_name"]}
        )
        outfile = Path(destdir) / f"{self.id}.th.nc"
        schism_ds.to_netcdf(outfile)
        return outfile


class SCHISMDataOcean(RompyBaseModel):
    elev2D: Optional[SCHISMDataBoundary] = Field(
        None,
        description="elev2D",
    )
    uv3D: Optional[SCHISMDataBoundary] = Field(
        None,
        description="uv3D",
    )
    TEM_3D: Optional[SCHISMDataBoundary] = Field(
        None,
        description="TEM_3D",
    )
    SAL_3D: Optional[SCHISMDataBoundary] = Field(
        None,
        description="SAL_3D",
    )

    @model_validator(mode="after")
    def not_yet_implemented(cls, v):
        for variable in ["uv3D", "TEM_3D", "SAL_3D"]:
            if getattr(v, variable) is not None:
                raise NotImplementedError(f"Variable {variable} is not yet implemented")
        return v

    @model_validator(mode="after")
    def set_id(cls, v):
        for variable in ["elev2D", "uv3D", "TEM_3D", "SAL_3D"]:
            if getattr(v, variable) is not None:
                getattr(v, variable).id = variable
        return v

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid2D | SCHISMGrid3D,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write all inputs to netcdf files.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid2D | SCHISMGrid3D
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        for variable in ["elev2D", "uv3D", "TEM_3D", "SAL_3D"]:
            data = getattr(self, variable)
            if data is None:
                continue
            data.get(destdir, grid, time)

    def __str__(self):
        return f"SCHISMDataOcean"
