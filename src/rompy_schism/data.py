from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from pydantic import ConfigDict, Field, model_validator

from rompy.core.boundary import BoundaryWaveStation, DataBoundary
from rompy.core.data import DataBlob, DataGrid
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel
from rompy.formatting import ARROW
from rompy.logging import get_logger
from rompy_schism.boundary_core import (
    BoundaryHandler,
    ElevationType,
    TidalDataset,
    TracerType,
    VelocityType,
)
from rompy_schism.grid import SCHISMGrid
from rompy_schism.tides_enhanced import BoundarySetup
from rompy.utils import total_seconds

from .namelists import Sflux_Inputs

logger = get_logger(__name__)


# Boundary files written from data: id, source attribute and which setups use them
BOUNDARY_FILES = {
    "elev2D": (
        "elev_source",
        lambda s: (
            s.elev_type in (ElevationType.EXTERNAL, ElevationType.HARMONICEXTERNAL)
        ),
    ),
    "uv3D": (
        "vel_source",
        lambda s: (
            s.vel_type
            in (
                VelocityType.EXTERNAL,
                VelocityType.HARMONICEXTERNAL,
                VelocityType.RELAXED,
            )
        ),
    ),
    "TEM_3D": ("temp_source", lambda s: s.temp_type == TracerType.EXTERNAL),
    "SAL_3D": ("salt_source", lambda s: s.salt_type == TracerType.EXTERNAL),
}


def to_python_type(value):
    """Convert numpy types to Python native types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    else:
        return value


class SfluxSource(DataGrid):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux"] = Field(
        default="sflux",
        description="Model type discriminator",
    )
    id: str = Field(default="sflux_source", description="id of the source")
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
    time_buffer: list[int] = Field(
        default=[0, 1],
        description="Number of source data timesteps to buffer the time range if `filter_time` is True",
    )
    # The source field needs special handling
    source: Any = None
    _variable_names = []

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    def __init__(self, **data):
        # Special handling for the DataGrid source field
        # Pydantic v2 is strict about union tag validation, so we need to handle it manually
        source_obj = None
        if "source" in data:
            source_obj = data.pop("source")  # Remove source to avoid validation errors

        # Initialize without the source field
        try:
            super().__init__(**data)
            # Set the source object after initialization
            if source_obj is not None:
                self.source = source_obj
        except Exception as e:
            logger.error(f"Error initializing SfluxSource: {e}")
            logger.error(f"Input data: {data}")
            raise

        # Initialize variable names
        self._set_variables()

    @property
    def outfile(self) -> str:
        # TODO - filenumber is. Hardcoded to 1 for now.
        return f"{self.id}.{str(1).rjust(4, '0')}.nc"

    def _set_variables(self) -> None:
        for variable in self._variable_names:
            if getattr(self, variable) is not None:
                self.variables.append(getattr(self, variable))

    @property
    def namelist(self) -> dict:
        # ret = self.model_dump()
        ret = {}
        for key, value in self.model_dump().items():
            if key in ["relative_weight", "max_window_hours", "fail_if_missing"]:
                ret.update({f"{self.id}_{key}": value})
        for varname in self._variable_names:
            var = getattr(self, varname)
            if var is not None:
                ret.update({varname: var})
            else:
                ret.update({varname: varname.replace("_name", "")})
        ret.update({f"{self.id}_file": self.id})
        return ret

    @property
    def ds(self):
        """Return the xarray dataset for this data source."""
        ds = self.source.open(
            variables=self.variables, filters=self.filter, coords=self.coords
        )
        # Define a dictionary for potential renaming
        rename_dict = {self.coords.y: "ny_grid", self.coords.x: "nx_grid"}

        # Construct a valid renaming dictionary
        valid_rename_dict = get_valid_rename_dict(ds, rename_dict)

        # Perform renaming if necessary
        if valid_rename_dict:
            ds = ds.rename_dims(valid_rename_dict)

        lon, lat = np.meshgrid(ds[self.coords.x], ds[self.coords.y])
        ds["lon"] = (("ny_grid", "nx_grid"), lon)
        ds["lat"] = (("ny_grid", "nx_grid"), lat)
        basedate = pd.to_datetime(ds.time.values[0])
        unit = f"days since {basedate.strftime('%Y-%m-%d %H:%M:%S')}"
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
            # "units": unit,
        }
        ds.time.encoding["units"] = unit
        ds.time.encoding["calendar"] = "proleptic_gregorian"
        # open bad dataset

        # SCHISM doesn't like scale_factor and add_offset attributes and requires Float64 values
        for var in ds.data_vars:
            # If the variable has scale_factor or add_offset attributes, remove them
            if "scale_factor" in ds[var].encoding:
                del ds[var].encoding["scale_factor"]
            if "add_offset" in ds[var].encoding:
                del ds[var].encoding["add_offset"]
            # set the data variable encoding to Float64
            ds[var].encoding["dtype"] = np.dtypes.Float64DType()

        return ds


class SfluxAir(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_air"] = Field(
        default="sflux_air",
        description="Model type discriminator",
    )
    uwind_name: Optional[str] = Field(
        None,
        description="name of zonal wind variable in source",
    )
    vwind_name: Optional[str] = Field(
        None,
        description="name of meridional wind variable in source",
    )
    prmsl_name: Optional[str] = Field(
        None,
        description="name of mean sea level pressure variable in source",
    )
    stmp_name: Optional[str] = Field(
        None,
        description="name of surface air temperature variable in source",
    )
    spfh_name: Optional[str] = Field(
        None,
        description="name of specific humidity variable in source",
    )

    # Allow extra fields during validation but exclude them from the model
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Allow extra fields during validation
        populate_by_name=True,  # Enable population by field name
    )

    def __init__(self, **data):
        # Initialize logger at the beginning

        # Pre-process parameters before passing to pydantic
        # Map parameters without _name suffix to ones with suffix
        name_mappings = {
            "uwind": "uwind_name",
            "vwind": "vwind_name",
            "prmsl": "prmsl_name",
            "stmp": "stmp_name",
            "spfh": "spfh_name",
        }

        for old_name, new_name in name_mappings.items():
            if old_name in data and new_name not in data:
                data[new_name] = data.pop(old_name)

        # Extract source to handle it separately (avoiding validation problems)
        source_obj = None
        if "source" in data:
            source_obj = data.pop("source")  # Remove source to avoid validation errors

            # Import here to avoid circular import
            from rompy.core.source import SourceFile, SourceIntake

            # If source is a dictionary, convert it to a proper source object
            if isinstance(source_obj, dict):
                logger.info(
                    f"Converting source dictionary to source object: {source_obj}"
                )

                # Handle different source types based on what's in the dictionary
                if "uri" in source_obj:
                    # Create a SourceFile or SourceIntake based on the URI
                    uri = source_obj["uri"]
                    if uri.startswith("intake://") or uri.endswith(".yaml"):
                        source_obj = SourceIntake(uri=uri)
                    else:
                        source_obj = SourceFile(uri=uri)
                    logger.info(f"Created source object from URI: {uri}")
                else:
                    raise ValueError(f"SfluxAir source needs a 'uri', got {source_obj}")
        else:
            raise ValueError("SfluxAir requires a 'source' parameter")

        # Call the parent constructor with the processed data (without source)
        try:
            super().__init__(**data)
        except Exception as e:
            logger.error(f"Error initializing SfluxAir: {e}")
            logger.error(f"Input data: {data}")
            raise

        # Set source manually after initialization
        self.source = source_obj
        logger.info(
            f"Successfully created SfluxAir instance with source type: {type(self.source)}"
        )

    _variable_names = [
        "uwind_name",
        "vwind_name",
        "prmsl_name",
        "stmp_name",
        "spfh_name",
    ]

    # Values for variables missing from the source: a standard atmosphere
    _missing_values = {
        "uwind_name": 0.0,  # m/s
        "vwind_name": 0.0,  # m/s
        "prmsl_name": 101325.0,  # Pa, SCHISM's prmsl_ref
        "stmp_name": 288.15,  # K
        "spfh_name": 0.01,  # kg/kg
    }

    @property
    def ds(self):
        """Return the xarray dataset for this data source.

        SCHISM needs all five sflux air variables. Those without a variable name are
        filled with standard-atmosphere values, so a wind-only source gives uniform
        pressure (no pressure gradient) and a plausible air density.
        """
        ds = super().ds
        template = next(
            (
                ds[getattr(self, name)]
                for name in self._variable_names
                if getattr(self, name) is not None
            ),
            None,
        )
        if template is None:
            raise ValueError("SfluxAir needs the name of at least one variable")
        filled = []
        for variable in self._variable_names:
            if getattr(self, variable) is None:
                proxy_var = variable.replace("_name", "")
                ds[proxy_var] = xr.full_like(
                    template, self._missing_values[variable], dtype="float64"
                )
                ds[proxy_var].attrs = {"long_name": proxy_var}
                filled.append(f"{proxy_var}={self._missing_values[variable]}")
        if filled:
            logger.info(f"  • Constant sflux air variables: {', '.join(filled)}")
        return ds


class SfluxRad(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_rad"] = Field(
        default="sflux_rad",
        description="Model type discriminator",
    )
    dlwrf_name: str = Field(
        None,
        description="name of downward long wave radiation variable in source",
    )
    dswrf_name: str = Field(
        None,
        description="name of downward short wave radiation variable in source",
    )
    _variable_names = ["dlwrf_name", "dswrf_name"]


class SfluxPrc(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_prc"] = Field(
        default="sflux_prc",
        description="Model type discriminator",
    )
    prate_name: str = Field(
        None,
        description="name of precipitation rate variable in source",
    )
    _variable_names = ["prate_name"]


class SCHISMDataSflux(RompyBaseModel):
    data_type: Literal["sflux"] = Field(
        default="sflux",
        description="Model type discriminator",
    )
    air_1: Optional[Any] = Field(None, description="sflux air source 1")
    air_2: Optional[Any] = Field(None, description="sflux air source 2")
    rad_1: Optional[Union[DataBlob, SfluxRad]] = Field(
        None, description="sflux rad source 1"
    )
    rad_2: Optional[Union[DataBlob, SfluxRad]] = Field(
        None, description="sflux rad source 2"
    )
    prc_1: Optional[Union[DataBlob, SfluxPrc]] = Field(
        None, description="sflux prc source 1"
    )
    prc_2: Optional[Union[DataBlob, SfluxPrc]] = Field(
        None, description="sflux prc source 2"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    def __init__(self, **data):
        # Handle 'air' parameter by mapping it to 'air_1'
        if "air" in data:
            air_value = data.pop("air")

            # If air is a dict, convert it to a SfluxAir instance
            if isinstance(air_value, dict):
                try:
                    # Import here to avoid circular import
                    from rompy_schism.data import SfluxAir

                    air_value = SfluxAir(**air_value)
                    logger.info(
                        "Successfully created SfluxAir instance from dictionary"
                    )
                except Exception as e:
                    logger.error(f"Failed to create SfluxAir instance: {e}")
                    # Fall back to passing the dictionary directly
                    logger.info(f"Falling back to dictionary: {air_value}")

            data["air_1"] = air_value

        # Call the parent constructor with the processed data
        super().__init__(**data)

    @model_validator(mode="after")
    def validate_air_fields(self):
        """Validate air fields after model creation."""
        # Convert dictionary to SfluxAir if needed
        if isinstance(self.air_1, dict):
            try:
                # Import here to avoid circular import
                from rompy_schism.data import SfluxAir

                logger.info(
                    f"Converting air_1 dictionary to SfluxAir object: {self.air_1}"
                )
                self.air_1 = SfluxAir(**self.air_1)
                logger.info("Successfully converted air_1 to SfluxAir instance")
            except Exception as e:
                logger.error(f"Error converting air_1 dictionary to SfluxAir: {e}")
                logger.error(f"Input data: {self.air_1}")
                # We'll let validation continue with the dictionary

        if isinstance(self.air_2, dict):
            try:
                from rompy_schism.data import SfluxAir

                logger.info(
                    f"Converting air_2 dictionary to SfluxAir object: {self.air_2}"
                )
                self.air_2 = SfluxAir(**self.air_2)
                logger.info("Successfully converted air_2 to SfluxAir instance")
            except Exception as e:
                logger.error(f"Error converting air_2 dictionary to SfluxAir: {e}")
                logger.error(f"Input data: {self.air_2}")

        return self

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
        ret = {}
        destdir = Path(destdir) / "sflux"
        destdir.mkdir(parents=True, exist_ok=True)
        namelistargs = {}

        # Collect information about active variables for logging
        active_variables = []
        source_info = {}

        # SCHISM needs forcing beyond the run: one day on each side (once, for all
        # variables)
        if time is not None:
            time = TimeRange(
                start=time.start - pd.Timedelta(days=1),
                end=time.end + pd.Timedelta(days=1),
                interval=time.interval,
            )

        for variable in ["air_1", "air_2", "rad_1", "rad_2", "prc_1", "prc_2"]:
            data = getattr(self, variable)
            if data is None:
                continue
            data.id = variable
            active_variables.append(variable)

            # Get source information
            if hasattr(data, "source") and hasattr(data.source, "uri"):
                source_info[variable] = str(data.source.uri)

            logger.debug(f"Processing {variable}")
            namelistargs.update(data.namelist)
            ret[variable] = data.get(destdir, grid, time)

        # Log summary of atmospheric data processing
        if active_variables:
            logger.info(f"  • Variables: {', '.join(active_variables)}")
            if source_info:
                unique_sources = list(set(source_info.values()))
                if len(unique_sources) == 1:
                    logger.info(f"  • Source: {unique_sources[0]}")
                else:
                    logger.info(f"  • Sources: {len(unique_sources)} files")
            logger.info(f"  • Output: {destdir}")

        ret["nml"] = Sflux_Inputs(**namelistargs).write_nml(destdir)
        return ret

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
                weight += data.relative_weight
                active = True
            if active and weight != 1.0:
                raise ValueError(
                    f"Relative weights for {variable} do not add to 1.0: {weight}"
                )
        return v


class SCHISMDataWave(BoundaryWaveStation):
    """This class is used to write wave spectral boundary data. Spectral data is extracted
    from the nearest points along the grid boundary"""

    data_type: Literal["wave"] = Field(
        default="wave",
        description="Model type discriminator",
    )
    sel_method: Literal["idw", "nearest"] = Field(
        default="nearest",
        description="Method for selecting boundary points",
    )
    sel_method_kwargs: dict = Field(
        default={"unique": True},
        description="Keyword arguments for sel_method",
    )
    time_buffer: list[int] = Field(
        default=[0, 1],
        description="Number of source data timesteps to buffer the time range if `filter_time` is True",
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
        logger.debug(f"Processing wave data: {self.id}")
        if self.crop_data and time is not None:
            self._filter_time(time)
        ds = self._sel_boundary(grid)
        outfile = Path(destdir) / f"{self.id}.nc"
        ds.spec.to_ww3(outfile)
        logger.debug(f"Saved wave data to {outfile}")
        return outfile

    @property
    def ds(self):
        """Return the filtered xarray dataset instance."""
        ds = super().ds
        for var in ds.data_vars:
            # If the variable has scale_factor or add_offset attributes, remove them
            if "scale_factor" in ds[var].encoding:
                del ds[var].encoding["scale_factor"]
            if "add_offset" in ds[var].encoding:
                del ds[var].encoding["add_offset"]
            # set the data variable encoding to Float64
            ds[var].encoding["dtype"] = np.dtypes.Float64DType()
        return ds

    def __str__(self):
        return "SCHISMDataWave"


class SCHISMDataBoundary(DataBoundary):
    """This class is used to extract ocean boundary data from a gridded dataset at the
    nodes of the open boundaries, all of them unless ``open_boundaries`` is set."""

    data_type: Literal["boundary"] = Field(
        default="boundary",
        description="Model type discriminator",
    )
    id: str = Field(
        "bnd",
        description="SCHISM th id of the source",
        json_schema_extra={"choices": ["elev2D", "uv3D", "TEM_3D", "SAL_3D", "bnd"]},
    )

    # This field is used to handle DataGrid sources in Pydantic v2
    data_grid_source: Optional[DataGrid] = Field(
        None, description="DataGrid source for boundary data"
    )
    variables: list[str] = Field(
        default_factory=list, description="variable name in the dataset"
    )
    sel_method: Literal["sel", "interp"] = Field(
        default="interp",
        description=(
            "Xarray method to use for selecting boundary points from the dataset"
        ),
    )
    time_buffer: list[int] = Field(
        default=[0, 1],
        description="Number of source data timesteps to buffer the time range if `filter_time` is True",
    )
    open_boundaries: Optional[list[int]] = Field(
        None,
        description=(
            "Indices of the open boundaries whose nodes are written, in mesh order; "
            "all when not set. SCHISMDataBoundaryConditions sets it to the "
            "boundaries that use the file."
        ),
    )

    def _boundary_nodes(self, grid: SCHISMGrid) -> list[int]:
        """Mesh nodes of the selected open boundaries, in bctides.in order."""
        gd = grid.pylibs_hgrid
        if hasattr(gd, "compute_bnd") and not hasattr(gd, "nob"):
            gd.compute_bnd()
        if not getattr(gd, "nob", 0):
            raise ValueError("No open boundary nodes found in the grid")
        indices = (
            range(gd.nob) if self.open_boundaries is None else self.open_boundaries
        )
        return [node for i in indices for node in gd.iobn[i]]

    def _boundary_points(self, grid: SCHISMGrid) -> tuple:
        """Coordinates of the selected open boundary nodes."""
        nodes = self._boundary_nodes(grid)
        return grid.pylibs_hgrid.x[nodes], grid.pylibs_hgrid.y[nodes]

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
        outfile = Path(destdir) / f"{self.id}.th.nc"
        boundary_ds = self.boundary_ds(grid, time)
        boundary_ds.to_netcdf(outfile, "w", "NETCDF3_CLASSIC", unlimited_dims="time")

        # Log file details with dimensions
        if "time_series" in boundary_ds.data_vars:
            shape = boundary_ds.time_series.shape
            logger.debug(f"Saved {self.id} to {outfile} (shape: {shape})")
        else:
            logger.debug(f"Saved boundary data to {outfile}")
        return outfile

    def boundary_ds(self, grid: SCHISMGrid, time: Optional[TimeRange]) -> xr.Dataset:
        """Generate SCHISM boundary dataset from source data.

        This function extracts and formats boundary data for SCHISM from a source dataset.
        For 3D models, it handles vertical interpolation to the SCHISM sigma levels.

        Parameters
        ----------
        grid : SCHISMGrid
            The SCHISM grid to extract boundary data for
        time : Optional[TimeRange]
            The time range to filter data to, if crop_data is True

        Returns
        -------
        xr.Dataset
            Dataset formatted for SCHISM boundary input
        """
        logger.debug(f"Fetching {self.id}")
        if self.crop_data and time is not None:
            self._filter_time(time)

        # Extract boundary data from source
        ds = self._sel_boundary(grid)

        # Calculate time step
        if len(ds.time) > 1:
            dt = total_seconds((ds.time[1] - ds.time[0]).values)
        else:
            dt = 3600

        # Get the variable data - handle multiple variables (e.g., u,v for velocity)
        num_components = len(self.variables)

        # Process all variables and stack them
        variable_data = []
        for var in self.variables:
            variable_data.append(ds[var].values)

        # Stack variables along a new component axis (last axis)
        if num_components == 1:
            data = variable_data[0]
        else:
            data = np.stack(variable_data, axis=-1)

        # Determine if we're working with 3D data
        is_3d_data = grid.is_3d and self.coords.z is not None

        # Handle different data dimensions based on 2D or 3D
        if is_3d_data:
            # Try to determine the dimension order
            if hasattr(ds[self.variables[0]], "dims"):
                # Get dimension names
                dims = list(ds[self.variables[0]].dims)

                # Find indices of time, z, and x dimensions
                time_dim_idx = dims.index(ds.time.dims[0])
                z_dim_idx = (
                    dims.index(ds[self.coords.z].dims[0])
                    if self.coords and self.coords.z and self.coords.z in ds
                    else 1
                )
                x_dim_idx = (
                    dims.index(ds[self.coords.x].dims[0])
                    if self.coords and self.coords.x and self.coords.x in ds
                    else 2
                )

                logger.debug(
                    f"Dimension order: time={time_dim_idx}, z={z_dim_idx}, x={x_dim_idx}"
                )

                # Reshape data to expected format if needed (time, x, z, [components])
                if num_components == 1:
                    # Single component case - need to transpose to (time, x, z)
                    if not (time_dim_idx == 0 and x_dim_idx == 1 and z_dim_idx == 2):
                        trans_dims = list(range(data.ndim))
                        trans_dims[time_dim_idx] = 0
                        trans_dims[x_dim_idx] = 1
                        trans_dims[z_dim_idx] = 2

                        data = np.transpose(data, trans_dims)
                        logger.debug(f"Transposed data shape: {data.shape}")

                    # Add the component dimension for SCHISM
                    time_series = np.expand_dims(data, axis=3)
                else:
                    # Multiple component case - data is already (time, x, z, components)
                    # Need to transpose the first 3 dimensions to (time, x, z) if needed
                    if not (time_dim_idx == 0 and x_dim_idx == 1 and z_dim_idx == 2):
                        trans_dims = list(
                            range(data.ndim - 1)
                        )  # Exclude component axis
                        trans_dims[time_dim_idx] = 0
                        trans_dims[x_dim_idx] = 1
                        trans_dims[z_dim_idx] = 2
                        # Keep component axis at the end
                        trans_dims.append(data.ndim - 1)

                        data = np.transpose(data, trans_dims)
                        logger.debug(f"Transposed data shape: {data.shape}")

                    # Data already has component dimension from stacking
                    time_series = data
            else:
                # Fallback: add component dimension if needed
                if num_components == 1:
                    time_series = np.expand_dims(data, axis=3)
                else:
                    time_series = data

            # Calculate zcor for 3D
            # For PyLibs vgrid, extract sigma coordinates differently
            gd = grid.pylibs_hgrid
            vgd = grid.pylibs_vgrid

            # Nodes of the selected open boundaries
            boundary_indices = self._boundary_nodes(grid)

            # Get bathymetry for boundary nodes
            boundary_depths = gd.dp[boundary_indices]

            # Get sigma levels from vgrid
            # Note: This assumes a simple sigma or SZ grid format
            # For more complex vgrids, more sophisticated extraction would be needed
            if vgd is not None:
                if hasattr(vgd, "sigma"):
                    sigma_levels = vgd.sigma.copy()
                    num_sigma_levels = len(sigma_levels)
                else:
                    # Default sigma levels if not available
                    sigma_levels = np.array([-1.0, 0.0])
                    num_sigma_levels = 2

                # Get fixed z levels if available
                if hasattr(vgd, "ztot"):
                    z_levels = vgd.ztot
                else:
                    z_levels = np.array([])

            # For each boundary point, determine the total number of vertical levels
            # and create appropriate zcor arrays
            all_zcors = []
            all_nvrt = []

            for i, (node_idx, depth) in enumerate(
                zip(boundary_indices, boundary_depths)
            ):
                # Check if we're in deep water (depth > first z level)
                if z_levels.size > 0 and depth > z_levels[0]:
                    # In deep water, find applicable z levels (between first z level and actual depth)
                    first_z_level = z_levels[0]
                    z_mask = (z_levels > first_z_level) & (z_levels < depth)
                    applicable_z = z_levels[z_mask] if np.any(z_mask) else []

                    # Total levels = sigma levels + applicable z levels
                    total_levels = num_sigma_levels + len(applicable_z)

                    # Create zcor for this boundary point
                    node_zcor = np.zeros(total_levels)

                    # First, calculate sigma levels using the first z level as the "floor"
                    for j in range(num_sigma_levels):
                        node_zcor[j] = first_z_level * sigma_levels[j]

                    # Then, add the fixed z levels below the sigma levels
                    for j, z_val in enumerate(applicable_z):
                        node_zcor[num_sigma_levels + j] = z_val

                else:
                    # In shallow water, just use sigma levels scaled to the actual depth
                    total_levels = num_sigma_levels

                    # Create zcor for this boundary point
                    node_zcor = np.zeros(total_levels)

                    for j in range(total_levels):
                        node_zcor[j] = depth * sigma_levels[j]

                # Store this boundary point's zcor and number of levels
                all_zcors.append(node_zcor)
                all_nvrt.append(total_levels)

            # Now we have a list of zcor arrays with potentially different lengths
            # Find the maximum number of levels across all boundary points
            max_nvrt = max(all_nvrt) if all_nvrt else num_sigma_levels

            # Create a uniform zcor array with the maximum number of levels
            zcor = np.zeros((len(boundary_indices), max_nvrt))

            # Fill in the values, leaving zeros for levels beyond a particular boundary point's total
            for i, (node_zcor, nvrt_i) in enumerate(zip(all_zcors, all_nvrt)):
                zcor[i, :nvrt_i] = node_zcor

            # Get source z-levels and prepare for interpolation
            sigma_values = (
                ds[self.coords.z].values
                if self.coords and self.coords.z
                else np.array([0])
            )
            data_shape = time_series.shape

            # Initialize interpolated data array with the maximum number of vertical levels
            if num_components == 1:
                interpolated_data = np.zeros((data_shape[0], data_shape[1], max_nvrt))
            else:
                interpolated_data = np.zeros(
                    (data_shape[0], data_shape[1], max_nvrt, data_shape[3])
                )

            # For each time step and boundary point
            for t in range(data_shape[0]):  # time
                for n in range(data_shape[1]):  # boundary points
                    # Get z-coordinates for this point
                    z_dest = zcor[n, :]
                    nvrt_n = all_nvrt[
                        n
                    ]  # Get the number of vertical levels for this point

                    if num_components == 1:
                        # Extract vertical profile for single component
                        profile = time_series[t, n, :, 0]

                        # Create interpolator for this profile
                        interp = sp.interpolate.interp1d(
                            sigma_values,
                            profile,
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )

                        # Interpolate to SCHISM levels for this boundary point
                        # Only interpolate up to the actual number of levels for this point
                        interpolated_data[t, n, :nvrt_n] = interp(z_dest[:nvrt_n])
                    else:
                        # Handle multiple components (e.g., u,v for velocity)
                        for c in range(num_components):
                            # Extract vertical profile for this component
                            profile = time_series[t, n, :, c]

                            # Create interpolator for this profile
                            interp = sp.interpolate.interp1d(
                                sigma_values,
                                profile,
                                kind="linear",
                                bounds_error=False,
                                fill_value="extrapolate",
                            )

                            # Interpolate to SCHISM levels for this boundary point
                            # Only interpolate up to the actual number of levels for this point
                            interpolated_data[t, n, :nvrt_n, c] = interp(
                                z_dest[:nvrt_n]
                            )

            # Replace data with interpolated values
            data = interpolated_data
            if num_components == 1:
                time_series = np.expand_dims(data, axis=3)
            else:
                time_series = data

            # Store the variable vertical levels in the output dataset
            # Create a 2D array where each row contains the vertical levels for a boundary node
            # For nodes with fewer levels, pad with NaN
            vert_levels = np.full((len(boundary_indices), max_nvrt), np.nan)
            for i, (node_zcor, nvrt_i) in enumerate(zip(all_zcors, all_nvrt)):
                vert_levels[i, :nvrt_i] = node_zcor

            # Create output dataset
            schism_ds = xr.Dataset(
                coords={
                    "time": ds.time,
                    "nOpenBndNodes": np.arange(time_series.shape[1]),
                    "nLevels": np.arange(max_nvrt),
                    "nComponents": np.arange(num_components),
                    "one": np.array([1]),
                },
                data_vars={
                    "time_step": (("one"), np.array([dt])),
                    "time_series": (
                        ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                        time_series,
                    ),
                    "vertical_levels": (
                        ("nOpenBndNodes", "nLevels"),
                        vert_levels,
                    ),
                    "num_levels": (
                        ("nOpenBndNodes"),
                        np.array(all_nvrt),
                    ),
                },
            )
        else:
            # # 2D case - simpler handling

            # Add level and component dimensions for SCHISM
            if num_components == 1:
                time_series = np.expand_dims(data, axis=(2, 3))
            else:
                # Multiple components: add level dimension but keep component dimension
                time_series = np.expand_dims(data, axis=2)

            # Create output dataset
            schism_ds = xr.Dataset(
                coords={
                    "time": ds.time,
                    "nOpenBndNodes": np.arange(time_series.shape[1]),
                    "nLevels": np.array([0]),  # Single level for 2D
                    "nComponents": np.arange(num_components),
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

        # Set attributes and encoding
        schism_ds.time_step.assign_attrs({"long_name": "time_step"})
        basedate = pd.to_datetime(ds.time.values[0])
        unit = f"days since {basedate.strftime('%Y-%m-%d %H:%M:%S')}"
        schism_ds.time.attrs = {
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
        }
        schism_ds.time.encoding["units"] = unit
        schism_ds.time.encoding["calendar"] = "proleptic_gregorian"

        # Handle missing values more robustly
        null_count = schism_ds.time_series.isnull().sum().item()
        if null_count > 0:
            logger.debug(
                f"Found {null_count} null values, applying interpolation and filling"
            )

            # Try interpolating along different dimensions
            for dim in ["nOpenBndNodes", "time", "nLevels"]:
                if dim in schism_ds.dims and len(schism_ds[dim]) > 1:
                    schism_ds["time_series"] = schism_ds.time_series.interpolate_na(
                        dim=dim
                    )
                    if not schism_ds.time_series.isnull().any():
                        logger.debug(
                            f"Interpolated missing values along {dim} dimension"
                        )
                        break

            # If still have NaNs, use more aggressive filling methods
            if schism_ds.time_series.isnull().any():
                # Find a reasonable fill value (median of non-NaN values)
                valid_values = schism_ds.time_series.values[
                    ~np.isnan(schism_ds.time_series.values)
                ]
                fill_value = np.median(valid_values) if len(valid_values) > 0 else 0.0
                schism_ds["time_series"] = schism_ds.time_series.fillna(fill_value)
                logger.debug(f"Filled remaining nulls with constant value {fill_value}")

        # Clean up encoding
        for var in schism_ds.data_vars:
            if "scale_factor" in schism_ds[var].encoding:
                del schism_ds[var].encoding["scale_factor"]
            if "add_offset" in schism_ds[var].encoding:
                del schism_ds[var].encoding["add_offset"]
            schism_ds[var].encoding["dtype"] = np.dtypes.Float64DType()

        return schism_ds


class SCHISMData(RompyBaseModel):
    """
    This class is used to gather all required input forcing for SCHISM
    """

    data_type: Literal["schism"] = Field(
        default="schism",
        description="Model type discriminator",
    )
    atmos: Optional[SCHISMDataSflux] = Field(None, description="atmospheric data")
    wave: Optional[Union[DataBlob, SCHISMDataWave]] = Field(
        None, description="wave data"
    )
    boundary_conditions: Optional["SCHISMDataBoundaryConditions"] = Field(
        None, description="unified boundary conditions (replaces tides and ocean)"
    )

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid,
        time: TimeRange,
    ) -> Dict[str, Any]:
        """
        Process all SCHISM forcing data and generate necessary input files.

        Parameters
        ----------
        destdir : str | Path
            Destination directory
        grid : SCHISMGrid
            SCHISM grid instance
        time : TimeRange
            Time range for the simulation

        Returns
        -------
        Dict[str, Any]
            Paths to generated files for each data component
        """
        from rompy.formatting import ARROW

        # Convert destdir to Path object
        destdir = Path(destdir)

        # Create destdir if it doesn't exist
        if not destdir.exists():
            destdir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Process atmospheric data
        if self.atmos:
            logger.info(f"{ARROW} Processing atmospheric forcing data")
            results["atmos"] = self.atmos.get(destdir, grid, time)
            logger.info(f"{ARROW} Atmospheric data processed successfully")

        # Process wave data
        if self.wave:
            logger.info(f"{ARROW} Processing wave boundary data")
            # Get source information
            if hasattr(self.wave, "source") and hasattr(self.wave.source, "uri"):
                logger.info(f"  • Source: {self.wave.source.uri}")
            elif hasattr(self.wave, "source") and hasattr(
                self.wave.source, "catalog_uri"
            ):
                logger.info(
                    f"  • Source: {self.wave.source.catalog_uri} (dataset: {getattr(self.wave.source, 'dataset_id', 'unknown')})"
                )

            results["wave"] = self.wave.get(destdir, grid, time)
            logger.info(f"  • Output: {results['wave']}")
            logger.info(f"{ARROW} Wave data processed successfully")

        # Process boundary conditions
        if self.boundary_conditions:
            logger.info(f"{ARROW} Processing boundary conditions")
            results["boundary_conditions"] = self.boundary_conditions.get(
                destdir, grid, time
            )
            logger.info(f"{ARROW} Boundary conditions processed successfully")

        return results

    def _format_value(self, obj):
        """Custom formatter for SCHISMData values.

        This method provides special formatting for specific types used in
        SCHISMData such as atmospheric, wave, and boundary data components.

        Args:
            obj: The object to format

        Returns:
            A formatted string or None to use default formatting
        """
        # Import specific types and formatting utilities
        from rompy.formatting import get_formatted_header_footer
        from rompy.logging import LoggingConfig

        # Get ASCII mode setting from LoggingConfig
        logging_config = LoggingConfig()
        USE_ASCII_ONLY = logging_config.use_ascii

        # Format SCHISMData (self-formatting)
        if isinstance(obj, SCHISMData):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM DATA CONFIGURATION", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Count and list data components
            components = {}
            if hasattr(obj, "atmos") and obj.atmos is not None:
                components["Atmospheric"] = type(obj.atmos).__name__
                # Add details for atmospheric data
                if hasattr(obj.atmos, "air_1") and obj.atmos.air_1 is not None:
                    air_sources = 1
                    if hasattr(obj.atmos, "air_2") and obj.atmos.air_2 is not None:
                        air_sources = 2
                    lines.append(f"      Air sources: {air_sources}")

                if hasattr(obj.atmos, "rad_1") and obj.atmos.rad_1 is not None:
                    rad_sources = 1
                    if hasattr(obj.atmos, "rad_2") and obj.atmos.rad_2 is not None:
                        rad_sources = 2
                    lines.append(f"      Radiation sources: {rad_sources}")

            if hasattr(obj, "wave") and obj.wave is not None:
                components["Wave"] = type(obj.wave).__name__

            if (
                hasattr(obj, "boundary_conditions")
                and obj.boundary_conditions is not None
            ):
                components["Boundary Conditions"] = type(
                    obj.boundary_conditions
                ).__name__

            for comp_name, comp_type in components.items():
                lines.append(f"  {bullet} {comp_name}: {comp_type}")

            if not components:
                lines.append(f"  {bullet} No data components configured")

            lines.append(footer)
            return "\n".join(lines)

        # Format SCHISMDataSflux
        if isinstance(obj, SCHISMDataSflux):
            header, footer, bullet = get_formatted_header_footer(
                title="ATMOSPHERIC DATA (SFLUX)", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Count air sources
            air_sources = 0
            if hasattr(obj, "air_1") and obj.air_1 is not None:
                air_sources += 1
            if hasattr(obj, "air_2") and obj.air_2 is not None:
                air_sources += 1

            if air_sources > 0:
                lines.append(f"  {bullet} Air sources: {air_sources}")

            # Count radiation sources
            rad_sources = 0
            if hasattr(obj, "rad_1") and obj.rad_1 is not None:
                rad_sources += 1
            if hasattr(obj, "rad_2") and obj.rad_2 is not None:
                rad_sources += 1

            if rad_sources > 0:
                lines.append(f"  {bullet} Radiation sources: {rad_sources}")

            # Check for precipitation
            if hasattr(obj, "prc_1") and obj.prc_1 is not None:
                lines.append(f"  {bullet} Precipitation: Available")

            lines.append(footer)
            return "\n".join(lines)

        # Format SCHISMDataWave
        if isinstance(obj, SCHISMDataWave):
            header, footer, bullet = get_formatted_header_footer(
                title="WAVE DATA", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            if hasattr(obj, "sel_method"):
                lines.append(f"  {bullet} Selection method: {obj.sel_method}")

            if hasattr(obj, "source") and obj.source is not None:
                source_type = type(obj.source).__name__
                lines.append(f"  {bullet} Source: {source_type}")

            lines.append(footer)
            return "\n".join(lines)

        # Format SCHISMDataBoundaryConditions
        if isinstance(obj, SCHISMDataBoundaryConditions):
            header, footer, bullet = get_formatted_header_footer(
                title="BOUNDARY CONDITIONS", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Count boundary setups
            boundary_count = 0
            if hasattr(obj, "boundaries") and obj.boundaries is not None:
                if isinstance(obj.boundaries, list):
                    boundary_count = len(obj.boundaries)
                else:
                    boundary_count = 1

            if boundary_count > 0:
                lines.append(f"  {bullet} Boundary setups: {boundary_count}")

            # Check for tidal components
            if hasattr(obj, "tidal") and obj.tidal is not None:
                lines.append(f"  {bullet} Tidal forcing: Available")

            lines.append(footer)
            return "\n".join(lines)

        # Use the new formatting framework
        from rompy.formatting import format_value

        return format_value(obj)


class HotstartConfig(RompyBaseModel):
    """
    Configuration for generating SCHISM hotstart files.

    This class specifies parameters for creating hotstart.nc files from
    temperature and salinity data sources already configured in boundary conditions.
    """

    enabled: bool = Field(
        default=False, description="Whether to generate hotstart file"
    )
    temp_var: str = Field(
        default="temperature",
        description="Name of temperature variable in source dataset",
    )
    salt_var: str = Field(
        default="salinity", description="Name of salinity variable in source dataset"
    )
    time_offset: float = Field(
        default=0.0, description="Offset to add to source time values (in days)"
    )
    time_base: datetime = Field(
        default=datetime(2000, 1, 1), description="Base time for source time values"
    )
    output_filename: str = Field(
        default="hotstart.nc", description="Name of the output hotstart file"
    )


class BoundarySetupWithSource(BoundarySetup):
    """
    Enhanced boundary setup that includes data sources.

    This class extends BoundarySetup to provide a unified configuration
    for both boundary conditions and their data sources.
    """

    elev_source: Optional[Union[DataBlob, DataGrid, SCHISMDataBoundary]] = Field(
        None, description="Data source for elevation boundary condition"
    )
    vel_source: Optional[Union[DataBlob, DataGrid, SCHISMDataBoundary]] = Field(
        None, description="Data source for velocity boundary condition"
    )
    temp_source: Optional[Union[DataBlob, DataGrid, SCHISMDataBoundary]] = Field(
        None, description="Data source for temperature boundary condition"
    )
    salt_source: Optional[Union[DataBlob, DataGrid, SCHISMDataBoundary]] = Field(
        None, description="Data source for salinity boundary condition"
    )

    @model_validator(mode="after")
    def validate_data_sources(self):
        """Ensure data sources are provided when needed for space-time boundary types."""
        # Check elevation data source
        if (
            self.elev_type in [ElevationType.EXTERNAL, ElevationType.HARMONICEXTERNAL]
            and self.elev_source is None
        ):
            logger.warning(
                "elev_source should be provided for EXTERNAL or HARMONICEXTERNAL elevation type"
            )

        # Check velocity data source
        if (
            self.vel_type
            in [
                VelocityType.EXTERNAL,
                VelocityType.HARMONICEXTERNAL,
                VelocityType.RELAXED,
            ]
            and self.vel_source is None
        ):
            logger.warning(
                "vel_source should be provided for EXTERNAL, HARMONICEXTERNAL, or RELAXED velocity type"
            )

        # Check temperature data source
        if self.temp_type == TracerType.EXTERNAL and self.temp_source is None:
            logger.warning(
                "temp_source should be provided for EXTERNAL temperature type"
            )

        # Check salinity data source
        if self.salt_type == TracerType.EXTERNAL and self.salt_source is None:
            logger.warning("salt_source should be provided for EXTERNAL salinity type")

        return self


class SCHISMDataBoundaryConditions(RompyBaseModel):
    """
    This class configures all boundary conditions for SCHISM including tidal,
    ocean, river, and nested model boundaries.

    It provides a unified interface for specifying boundary conditions and their
    data sources, replacing the separate tides and ocean configurations.
    """

    # Allow arbitrary types for schema generation
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_type: Literal["boundary_conditions"] = Field(
        default="boundary_conditions",
        description="Model type discriminator",
    )

    # Tidal dataset specification
    tidal_data: Optional[TidalDataset] = Field(
        None,
        description="Tidal forcing dataset",
    )

    # Boundary configurations with integrated data sources
    boundaries: Dict[int, BoundarySetupWithSource] = Field(
        default_factory=dict,
        description=(
            "Setup of open boundaries by their index in the mesh (0 is the first "
            "open boundary in hgrid.gr3)"
        ),
    )
    default_boundary: Optional[BoundarySetupWithSource] = Field(
        None,
        description="Setup of the open boundaries not listed in boundaries",
    )

    # Predefined configuration types
    setup_type: Optional[Literal["tidal", "hybrid", "river", "nested"]] = Field(
        None, description="Predefined boundary setup type"
    )

    # Hotstart configuration
    hotstart_config: Optional[HotstartConfig] = Field(
        None, description="Configuration for hotstart file generation"
    )

    @model_validator(mode="before")
    @classmethod
    def convert_numpy_types(cls, data):
        """Convert any numpy values to Python native types"""
        if not isinstance(data, dict):
            return data

        for key, value in list(data.items()):
            if isinstance(value, (np.bool_, np.integer, np.floating, np.ndarray)):
                data[key] = to_python_type(value)
        return data

    @model_validator(mode="after")
    def validate_tidal_data(self):
        """Ensure tidal data is provided when needed for TIDAL or TIDALSPACETIME boundaries."""
        setups = list((self.boundaries or {}).values())
        if self.default_boundary is not None:
            setups.append(self.default_boundary)
        needs_tidal_data = False

        # Check setup_type first
        if self.setup_type in ["tidal", "hybrid"]:
            needs_tidal_data = True

        # Then check individual boundaries
        for setup in setups:
            if (
                hasattr(setup, "elev_type")
                and setup.elev_type
                in [ElevationType.HARMONIC, ElevationType.HARMONICEXTERNAL]
            ) or (
                hasattr(setup, "vel_type")
                and setup.vel_type
                in [VelocityType.HARMONIC, VelocityType.HARMONICEXTERNAL]
            ):
                needs_tidal_data = True
                break

        if needs_tidal_data and not self.tidal_data:
            raise ValueError(
                "Tidal data is required for HARMONIC or HARMONICEXTERNAL boundary types but was not provided"
            )

        return self

    @model_validator(mode="after")
    def validate_setup_type(self):
        """Validate setup type specific requirements."""
        # Skip validation if setup_type is not set
        if not self.setup_type:
            return self

        if self.setup_type in ["tidal", "hybrid"]:
            if not self.tidal_data:
                raise ValueError(
                    "tidal_data is required for tidal or hybrid setup_type"
                )

        elif self.setup_type == "river":
            if self.boundaries:
                has_flow = any(
                    hasattr(s, "const_flow") and s.const_flow is not None
                    for s in self.boundaries.values()
                )
                if not has_flow:
                    raise ValueError(
                        "At least one boundary should have const_flow for river setup_type"
                    )

        elif self.setup_type == "nested":
            if self.boundaries:
                for idx, setup in self.boundaries.items():
                    if (
                        hasattr(setup, "vel_type")
                        and setup.vel_type == VelocityType.RELAXED
                    ):
                        if not hasattr(setup, "inflow_relax") or not hasattr(
                            setup, "outflow_relax"
                        ):
                            logger.warning(
                                f"inflow_relax and outflow_relax are recommended for nested setup_type in boundary {idx}"
                            )
        else:
            raise ValueError(
                f"Unknown setup_type: {self.setup_type}. Expected one of: tidal, hybrid, river, nested"
            )

        return self

    def boundary_setups(self, nob: int) -> Dict[int, BoundarySetupWithSource]:
        """Return the setup of each of the ``nob`` open boundaries of the mesh.

        SCHISM needs one entry per open boundary in bctides.in. A boundary takes
        its setup from ``boundaries``, or else from ``default_boundary``.
        """
        unknown = sorted(i for i in self.boundaries if not 0 <= i < nob)
        if unknown:
            raise ValueError(
                f"boundaries {unknown} are not open boundaries of the mesh, which "
                f"has {nob} (indices 0 to {nob - 1})"
            )
        setups = {i: self.boundaries.get(i, self.default_boundary) for i in range(nob)}
        missing = [i for i, setup in setups.items() if setup is None]
        if missing:
            raise ValueError(
                f"Open boundaries {missing} of the mesh have no setup: add them to "
                "boundaries, or set default_boundary"
            )
        return setups

    def _create_boundary_config(self, grid):
        """Create a TidalBoundary object based on the configuration."""
        # Get tidal data paths
        if self.tidal_data:
            if (
                hasattr(self.tidal_data, "tidal_database")
                and self.tidal_data.tidal_database
            ):
                str(self.tidal_data.tidal_database)

        # Ensure boundary information is computed
        if hasattr(grid.pylibs_hgrid, "compute_bnd"):
            grid.pylibs_hgrid.compute_bnd()
        else:
            logger.warning(
                "Grid object doesn't have compute_bnd method. Boundary information may be missing."
            )

        # Create a new TidalBoundary with all the configuration
        # Ensure boundary information is computed before creating the boundary
        if not hasattr(grid.pylibs_hgrid, "nob") or not hasattr(
            grid.pylibs_hgrid, "nobn"
        ):
            logger.info("Computing boundary information before creating TidalBoundary")
            # First try compute_bnd if available
            if hasattr(grid.pylibs_hgrid, "compute_bnd"):
                grid.pylibs_hgrid.compute_bnd()

            # Then try compute_all if nob is still missing
            if not hasattr(grid.pylibs_hgrid, "nob") and hasattr(
                grid.pylibs_hgrid, "compute_all"
            ):
                if hasattr(grid.pylibs_hgrid, "compute_all"):
                    grid.pylibs_hgrid.compute_all()

        # Verify boundary attributes are available
        if not hasattr(grid.pylibs_hgrid, "nob"):
            logger.error("Failed to set 'nob' attribute on grid.pylibs_hgrid")
            raise AttributeError(
                "Missing required 'nob' attribute on grid.pylibs_hgrid"
            )

        # Create TidalBoundary with pre-computed grid to avoid losing boundary info
        # Get the grid path for TidalBoundary
        grid_path = (
            str(grid.hgrid.path)
            if hasattr(grid, "hgrid") and hasattr(grid.hgrid, "path")
            else None
        )
        if grid_path is None:
            # Create a temporary file with the grid if needed
            import tempfile

            temp_file = tempfile.NamedTemporaryFile(suffix=".gr3", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            grid.pylibs_hgrid.write_hgrid(temp_path)
            grid_path = temp_path

        boundary = BoundaryHandler(
            grid_path=grid_path, tidal_data=self.tidal_data, nvrt=grid.nvrt or 2
        )

        # Replace the TidalBoundary's grid with our pre-computed one to preserve boundary info
        boundary.grid = grid.pylibs_hgrid

        # Configure each open boundary of the mesh
        for idx, setup in self.boundary_setups(grid.pylibs_hgrid.nob).items():
            boundary_config = setup.to_boundary_config()
            boundary.set_boundary_config(idx, boundary_config)

        return boundary

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid,
        time: TimeRange,
    ) -> Dict[str, str]:
        """
        Process all boundary data and generate necessary input files.

        Parameters
        ----------
        destdir : str | Path
            Destination directory
        grid : SCHISMGrid
            SCHISM grid instance
        time : TimeRange
            Time range for the simulation

        Returns
        -------
        Dict[str, str]
            Paths to generated files
        """
        # Processing boundary conditions

        # Convert destdir to Path object
        destdir = Path(destdir)

        # Create destdir if it doesn't exist
        if not destdir.exists():
            logger.info(f"Creating destination directory: {destdir}")
            destdir.mkdir(parents=True, exist_ok=True)

        # # 1. Process tidal data if needed
        if self.tidal_data:
            logger.info(
                f"{ARROW} Processing tidal constituents: {', '.join(self.tidal_data.constituents) if hasattr(self.tidal_data, 'constituents') else 'default'}"
            )
            self.tidal_data.get(grid)

        # 2. Create boundary condition file (bctides.in)
        boundary = self._create_boundary_config(grid)

        # Set start time and run duration
        start_time = time.start
        if time.end is not None and time.start is not None:
            run_days = (
                time.end - time.start
            ).total_seconds() / 86400.0  # Convert to days
        else:
            run_days = 1.0  # Default to 1 day if time is not properly specified
        boundary.set_run_parameters(start_time, run_days)

        # Generate bctides.in file
        bctides_path = destdir / "bctides.in"
        logger.info(f"{ARROW} Generating boundary condition file: bctides.in")

        # Ensure grid object has complete boundary information before writing
        if hasattr(grid.pylibs_hgrid, "compute_all"):
            grid.pylibs_hgrid.compute_all()

        # Double-check all required attributes are present
        required_attrs = ["nob", "nobn", "iobn"]
        missing_attrs = [
            attr
            for attr in required_attrs
            if not (grid.pylibs_hgrid and hasattr(grid.pylibs_hgrid, attr))
        ]
        if missing_attrs:
            error_msg = (
                f"Grid is missing required attributes: {', '.join(missing_attrs)}"
            )
            logger.error(error_msg)
            raise AttributeError(error_msg)

        # Write the boundary file - no fallbacks
        boundary.write_boundary_file(bctides_path)
        logger.info(f"{ARROW} Boundary conditions written successfully")

        # 3. Process ocean data based on boundary configurations
        processed_files = {"bctides": str(bctides_path)}

        # Boundary files: one per kind, with the nodes of the open boundaries that
        # use it, in mesh order, as SCHISM reads them
        setups = self.boundary_setups(grid.pylibs_hgrid.nob)
        for file_id, (attr, uses) in BOUNDARY_FILES.items():
            indices = [i for i, setup in setups.items() if uses(setup)]
            sources = [getattr(setups[i], attr) for i in indices]
            distinct = []
            for source in sources:
                if source not in distinct:
                    distinct.append(source)
            if not distinct or distinct == [None]:
                continue  # the file is provided some other way
            if len(distinct) > 1:
                raise ValueError(
                    f"{file_id}.th.nc holds the open boundaries {indices}, which "
                    f"use it, so they need the same {attr}"
                )
            source = distinct[0]
            logger.info(
                f"{ARROW} Writing {file_id}.th.nc for open boundaries {indices}"
            )
            if getattr(source, "data_type", None) == "boundary":
                source = source.model_copy(
                    update={"id": file_id, "open_boundaries": indices}
                )
                processed_files[file_id] = source.get(destdir, grid, time)
            else:
                # A file prepared elsewhere (DataBlob)
                processed_files[file_id] = source.get(str(destdir))

        # Generate hotstart file if configured
        if self.hotstart_config and self.hotstart_config.enabled:
            logger.info(f"{ARROW} Generating hotstart file")
            hotstart_path = self._generate_hotstart(destdir, grid, time)
            processed_files["hotstart"] = hotstart_path
            logger.info(f"  • Output: {hotstart_path}")

        # Log summary of processed files with more details
        boundary_data_files = [f for k, f in processed_files.items() if "boundary" in k]
        if boundary_data_files:
            logger.info(
                f"  • Files: {', '.join([Path(f).name for f in boundary_data_files])}"
            )

        return processed_files

    def _generate_hotstart(
        self,
        destdir: Union[str, Path],
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """
        Generate hotstart file using boundary condition data sources.

        Args:
            destdir: Destination directory for the hotstart file
            grid: SCHISM grid object
            time: Time range for the data

        Returns:
            Path to the generated hotstart file
        """
        from rompy_schism.hotstart import SCHISMDataHotstart

        # Find a boundary that has both temperature and salinity sources
        temp_source = None
        salt_source = None

        setups = list(self.boundaries.values())
        if self.default_boundary is not None:
            setups.append(self.default_boundary)
        for boundary_config in setups:
            if boundary_config.temp_source is not None:
                temp_source = boundary_config.temp_source
            if boundary_config.salt_source is not None:
                salt_source = boundary_config.salt_source

            # If we found both, we can proceed
            if temp_source is not None and salt_source is not None:
                break

        if temp_source is None or salt_source is None:
            raise ValueError(
                "Hotstart generation requires both temperature and salinity sources "
                "to be configured in boundary conditions"
            )

        # Create hotstart instance using the first available source
        # (assuming temp and salt sources point to the same dataset)
        # Include both temperature and salinity variables for hotstart generation
        temp_var_name = (
            self.hotstart_config.temp_var if self.hotstart_config else "temperature"
        )
        salt_var_name = (
            self.hotstart_config.salt_var if self.hotstart_config else "salinity"
        )

        # Log hotstart generation details
        logger.info(f"  • Variables: {temp_var_name}, {salt_var_name}")
        if hasattr(temp_source, "source") and hasattr(temp_source.source, "uri"):
            logger.info(f"  • Source: {temp_source.source.uri}")

        hotstart_data = SCHISMDataHotstart(
            source=temp_source.source,
            variables=[temp_var_name, salt_var_name],
            coords=getattr(temp_source, "coords", None),
            temp_var=temp_var_name,
            salt_var=salt_var_name,
            time_offset=(
                self.hotstart_config.time_offset if self.hotstart_config else 0.0
            ),
            time_base=(
                self.hotstart_config.time_base
                if self.hotstart_config
                else datetime(2000, 1, 1)
            ),
            output_filename=(
                self.hotstart_config.output_filename
                if self.hotstart_config
                else "hotstart.nc"
            ),
        )

        return hotstart_data.get(str(destdir), grid=grid, time=time)

    # def check_bctides_flags(cls, v):
    #     # TODO Add check fro bc flags in teh event of 3d inputs
    #     # SHould possibly move this these flags out of SCHISMDataTides class as they cover more than
    #     # just tides
    #     return cls


def get_valid_rename_dict(ds, rename_dict):
    """Construct a valid renaming dictionary that only includes names which need renaming."""
    valid_rename_dict = {}
    for old_name, new_name in rename_dict.items():
        if old_name in ds.dims and new_name not in ds.dims:
            valid_rename_dict[old_name] = new_name
    return valid_rename_dict
