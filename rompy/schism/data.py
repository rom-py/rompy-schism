import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union

import appdirs
import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import AnyPath
from pydantic import Field, field_validator, model_validator
from pyschism.forcing.bctides import Bctides

from rompy.core import DataGrid, RompyBaseModel
from rompy.core.boundary import (BoundaryWaveStation, DataBoundary, SourceFile,
                                 SourceWavespectra)
from rompy.core.data import DATA_SOURCE_TYPES, DataBlob
from rompy.core.time import TimeRange
from rompy.schism.grid import SCHISMGrid
from rompy.utils import total_seconds

from .namelists import Sflux_Inputs

logger = logging.getLogger(__name__)


class SfluxSource(DataGrid):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux"] = Field(
        default="sflux",
        description="Model type discriminator",
    )
    id: str = Field("sflux_source", description="id of the source")
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
    id: str = Field(None, description="id of the source", choices=["air", "rad", "prc"])

    @property
    def outfile(self) -> str:
        # TODO - filenumber is. Hardcoded to 1 for now.
        return f'{self.id}_1.{str(1).rjust(4, "0")}.nc'

    @property
    def namelist(self) -> dict:
        ret = self.model_dump()
        for key, value in self.model_dump().items():
            if key in ["relative_weight", "max_window_hours", "fail_if_missing"]:
                ret.update({f"{self.id}_{key}": value})
        ret.update({f"{self.id}_file": self.outfile})
        return ret

    @property
    def ds(self):
        """Return the xarray dataset for this data source."""
        ds = self.source.open(
            variables=self.variables, filters=self.filter, coords=self.coords
        )
        dt = total_seconds((ds.time[1] - ds.time[0]).values)
        times = np.arange(0, ds.time.size) * dt
        ds.time.assign_attrs({"long_name": "simulation_time"})
        basedate = pd.to_datetime(ds.time.values[0])
        ds["time"] = times
        ds.time.attrs = {
            "long_name": "Time",
            "standard_name": "time",
            "base_date": np.int32(
                np.array(
                    [
                        basedate.year,
                        basedate.month,
                        basedate.day,
                        basedate.hour,
                        basedate.minute,
                        basedate.second,
                    ]
                )
            ),
            "units": f"days since {basedate.strftime('%Y-%m-%d %H:%M:%S')}",
        }
        return ds


class SfluxAir(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_air"] = Field(
        default="sflux_air",
        description="Model type discriminator",
    )
    uwind_name: str = Field(
        "u10",
        description="name of zonal wind variable in source",
    )
    vwind_name: str = Field(
        "v10",
        description="name of meridional wind variable in source",
    )
    prmsl_name: str = Field(
        "mslp",
        description="name of mean sea level pressure variable in source",
    )
    stmp_name: str = Field(
        "stmp",
        description="name of surface air temperature variable in source",
    )
    spfh_name: SfluxSource = Field(
        "spfh",
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

    # @property
    # def namelist(self) -> dict:
    #     ret = super().namelist
    #     for key, value in self.model_dump().items():
    #         if key in [
    #             "uwind_name",
    #             "vwind_name",
    #             "prmsl_name",
    #             "stmp_name",
    #             "spfh_name",
    #         ]:
    #             ret.update({f"{self.id}_{key}": value})
    #     return ret


class SfluxRad(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_rad"] = Field(
        default="sflux_rad",
        description="Model type discriminator",
    )
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

    data_type: Literal["sflux_prc"] = Field(
        default="sflux_rad",
        description="Model type discriminator",
    )
    prate_name: SfluxSource = Field(
        None,
        description="name of precipitation rate variable in source",
    )

    def _set_variables(self) -> None:
        self.variables = [self.prate_name]


class SCHISMDataSflux(RompyBaseModel):
    data_type: Literal["sflux"] = Field(
        default="sflux",
        description="Model type discriminator",
    )
    air_1: SfluxAir = Field(None, description="sflux air source 1")
    air_2: SfluxAir = Field(None, description="sflux air source 2")
    rad_1: SfluxRad = Field(None, description="sflux rad source 1")
    rad_2: SfluxRad = Field(None, description="sflux rad source 2")
    prc_1: SfluxPrc = Field(None, description="sflux prc source 1")
    prc_2: SfluxPrc = Field(None, description="sflux prc source 2")

    def get(
        self,
        destdir: str | Path,
        grid: Optional[SCHISMGrid] = None,
        time: Optional[TimeRange] = None,
    ) -> Path:
        """Writes SCHISM sflux data from a dataset.

        Args:
            destdir (str | Path): The destination directory to write the sflux data.
            grid (Optional[SCHISMGrid], optional): The grid type. Defaults to None.
            time (Optional[TimeRange], optional): The time range. Defaults to None.

        Returns:
            Path: The path to the written sflux data.

        """
        destdir = Path(destdir) / "sflux"
        destdir.mkdir(parents=True, exist_ok=True)
        namelistargs = {}
        for variable in ["air_1", "air_2", "rad_1", "rad_2", "prc_1", "prc_2"]:
            data = getattr(self, variable)
            if data is None:
                continue
            data.id = variable
            logger.info(f"Fetching {variable}")
            namelistargs.update(data.namelist)
            namelistargs.update({f"{variable}_file": data.outfile})
            data.get(destdir, grid, time)
        Sflux_Inputs(**namelistargs).write_nml(destdir)

    @model_validator(mode="after")
    def check_weights(v):
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
            return v


class SCHISMDataWave(BoundaryWaveStation):
    """This class is used to write SCHISM data from a dataset."""

    data_type: Literal["wave"] = Field(
        default="wave",
        description="Model type discriminator",
    )
    sel_method: dict = Field(
        default="nearest",
        description="Keyword arguments for sel_method",
    )
    sel_method_kwargs: dict = Field(
        default={"unique": True},
        description="Keyword arguments for sel_method",
    )

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write the selected boundary data to a netcdf file.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        logger.info(f"Fetching {self.id}")
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

    data_type: Literal["boundary"] = Field(
        default="boundary",
        description="Model type discriminator",
    )
    id: str = Field(
        "bnd",
        description="SCHISM th id of the source",
        choices=["elev2D", "uv3D", "TEM_3D", "SAL_3D", "bnd"],
    )
    variable: str = Field(..., description="variable name in the dataset")
    sel_method: Literal["sel", "interp"] = Field(
        default="interp",
        description=(
            "Xarray method to use for selecting boundary points from the dataset"
        ),
    )

    @model_validator(mode="after")
    def _set_variables(self) -> "SCHISMDataBoundary":
        self.variables = [self.variable]
        return self

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write the selected boundary data to a netcdf file.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        # prepare xarray.Dataset and save forcing netCDF file
        logger.info(f"Fetching {self.id}")
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
        schism_ds.time_step.assign_attrs({"long_name": "time_step"})
        schism_ds.time.assign_attrs({"long_name": "simulation_time"})
        schism_ds.time_series.assign_attrs(
            {"long_name": ds[self.variable].attrs["long_name"]}
        )
        outfile = (
            Path(destdir) / f"{self.id}.th.nc"
        )  # the two is just a temporary fix to stop clash with tides
        schism_ds.to_netcdf(outfile)
        return outfile


class SCHISMDataOcean(RompyBaseModel):
    data_type: Literal["ocean"] = Field(
        default="ocean",
        description="Model type discriminator",
    )
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
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write all inputs to netcdf files.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid,
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


# def setup_bctides():
#     # Taken from example at https://schism-dev.github.io/schism/master/getting-started/pre-processing-with-pyschism/boundary.html I don't really understand this
#     # Ultimately these will not wanted to be hardcoded.
#     iet3 = iettype.Iettype3(constituents="major", database="tpxo")
#     iet4 = iettype.Iettype4()
#     iet5 = iettype.Iettype5(iettype3=iet3, iettype4=iet4)
#     ifl3 = ifltype.Ifltype3(constituents="major", database="tpxo")
#     ifl4 = ifltype.Ifltype4()
#     ifl5 = ifltype.Ifltype5(ifltype3=ifl3, ifltype4=ifl4)
#     # isa3 = isatype.Isatype4()
#     # ite3 = itetype.Itetype4()
#     return ifl5, iet5


class TidalDataset(RompyBaseModel):
    data_type: Literal["tidal_dataset"] = Field(
        default="tidal_dataset",
        description="Model type discriminator",
    )
    elevations: AnyPath = Field(..., description="Path to elevations file")
    velocities: AnyPath = Field(..., description="Path to currents file")

    def get(self, destdir: str | Path) -> str:
        """Write all inputs to netcdf files.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        # TODO need to put some smarts in here for remote files
        os.environ["TPXO_ELEVATION"] = self.elevations.as_posix()
        os.environ["TPXO_VELOCITY"] = self.velocities.as_posix()


class SCHISMDataTides(RompyBaseModel):
    data_type: Literal["tides"] = Field(
        default="tide",
        description="Model type discriminator",
    )
    tidal_data: TidalDataset = Field(..., description="tidal dataset")
    cutoff_depth: float = Field(
        50.0,
        description="cutoff depth for tides",
    )
    flags: Optional[list] = Field([[5, 5, 4, 4]], description="nested list of bctypes")
    constituents: Union[str, list] = Field("major", description="constituents")
    database: str = Field("tpxo", description="database", choices=["tpxo", "fes2014"])
    add_earth_tidal: bool = Field(True, description="add_earth_tidal")
    ethconst: Optional[list] = Field(
        [], description="constant elevation value for each open boundary"
    )
    vthconst: Optional[list] = Field(
        [], description="constant discharge value for each open boundary"
    )
    tthconst: Optional[list] = Field(
        [], description="constant temperature value for each open boundary"
    )
    sthconst: Optional[list] = Field(
        [], description="constant salinity value for each open boundary"
    )
    tobc: Optional[list[float]] = Field(
        [1], description="nuding factor of temperature for each open boundary"
    )
    sobc: Optional[list[float]] = Field(
        [1], description="nuding factor of salinity for each open boundary"
    )
    relax: Optional[list[float]] = Field(
        [], description="relaxation constants for inflow and outflow"
    )

    def get(self, destdir: str | Path, grid: SCHISMGrid, time: TimeRange) -> str:
        """Write all inputs to netcdf files.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """

        self.tidal_data.get(destdir)
        logger.info(f"Generating tides")
        bctides = Bctides(
            hgrid=grid.pyschism_hgrid,
            flags=self.flags,
            constituents=self.constituents,
            database=self.database,
            add_earth_tidal=self.add_earth_tidal,
            cutoff_depth=self.cutoff_depth,
            ethconst=self.ethconst,
            vthconst=self.vthconst,
            tthconst=self.tthconst,
            sthconst=self.sthconst,
            tobc=self.tobc,
            sobc=self.sobc,
            relax=self.relax,
        )
        bctides.write(
            destdir,  # +'/bctides.in',
            start_date=time.start,
            rnday=time.end - time.start,
            overwrite=True,
        )


class SCHISMData(RompyBaseModel):
    """
    This class is used to gather all required input forcing for SCHISM
    """

    data_type: Literal["schism"] = Field(
        default="schism",
        description="Model type discriminator",
    )
    atmos: Optional[SCHISMDataSflux] = Field(None, description="atmospheric data")
    ocean: Optional[SCHISMDataOcean] = Field(None, description="ocean data")
    wave: Optional[SCHISMDataWave] = Field(None, description="wave data")
    tides: Optional[SCHISMDataTides] = Field(None, description="tidal data")

    def get(
        self,
        destdir: str | Path,
        grid: Optional[SCHISMGrid] = None,
        time: Optional[TimeRange] = None,
    ) -> None:
        ret = {}
        for datatype in ["atmos", "ocean", "wave", "tides"]:
            data = getattr(self, datatype)
            if data is None:
                continue
            output = data.get(destdir, grid, time)
            ret.update({datatype: output})
        return ret
