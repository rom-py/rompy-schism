"""
Module for generating SCHISM hotstart files.

This module provides functionality to create hotstart.nc files for SCHISM
by interpolating temperature and salinity data from source datasets to the SCHISM grid.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field
from pylib import WriteNC, datenum, zdata

from rompy.core.data import DataGrid
from rompy.core.time import TimeRange
from rompy.logging import get_logger
from rompy_schism.fill import fill_below_seabed, fill_land
from rompy_schism.grid import SCHISMGrid

logger = get_logger(__name__)



class SCHISMDataHotstart(DataGrid):
    """
    This class is used to generate a hotstart file for SCHISM based on source data.

    It inherits from DataGrid and uses the source dataset to interpolate temperature
    and salinity data to the SCHISM grid and create a hotstart.nc file.
    """

    data_type: Literal["hotstart"] = Field(
        default="hotstart",
        description="Model type discriminator",
    )
    temp_var: str = Field(
        "water_temp", description="Name of temperature variable in source dataset"
    )
    salt_var: str = Field(
        "salinity", description="Name of salinity variable in source dataset"
    )
    time_offset: float = Field(
        0.0, description="Offset to add to source time values (in days)"
    )
    time_base: datetime = Field(
        datetime(2000, 1, 1), description="Base time for source time values"
    )
    output_filename: str = Field(
        "hotstart.nc", description="Name of the output hotstart file"
    )

    def get(
        self,
        destdir: Union[str, Path],
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """
        Generate a hotstart file for SCHISM based on source data.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the hotstart file.
        grid : SCHISMGrid
            SCHISM grid to interpolate data to.
        time : Optional[TimeRange]
            Time range for the data (not used, as hotstart is for a single time).

        Returns
        -------
        str
            Path to the generated hotstart file.
        """
        logger.debug(f"Generating hotstart file in {destdir}")
        destdir = Path(destdir)
        destdir.mkdir(parents=True, exist_ok=True)
        output_path = destdir / self.output_filename

        target_time = time.start

        # Convert to datenum format for pylibs
        start_t = datenum(
            target_time.year,
            target_time.month,
            target_time.day,
            target_time.hour,
            target_time.minute,
            target_time.second,
        )

        # Get dataset directly using DataGrid's ds() method
        ds = self.ds

        # Find the closest time in the dataset
        if self.coords.t in ds.dims or self.coords.t in ds.coords:
            time_values = ds[self.coords.t].values
            if np.issubdtype(time_values.dtype, np.datetime64):
                time_values = np.array(
                    [
                        datenum(t.year, t.month, t.day, t.hour, t.minute, t.second)
                        for t in pd.to_datetime(time_values)
                    ]
                )
            else:
                # Numeric values: hours since time_base, plus time_offset days
                time_values = (
                    time_values.astype(float) / 24.0
                    + datenum(
                        self.time_base.year,
                        self.time_base.month,
                        self.time_base.day,
                    )
                    + self.time_offset
                )

            # Find closest time index
            time_idx = np.argmin(np.abs(time_values - start_t))
            logger.debug(
                f"Using time index {time_idx} (closest to requested start time)"
            )

            # Select the data at this time
            if self.coords.t in ds.dims:
                ds = ds.isel({self.coords.t: time_idx})
        else:
            logger.warning(
                f"Time variable '{self.coords.t}' not found in dataset. Using all data."
            )

        # Read grid information
        # We need to access the pylibs grid objects directly
        if not hasattr(grid, "pylibs_hgrid") or grid.pylibs_hgrid is None:
            # Load the grid if not already loaded
            grid.load()

        # Get grid dimensions from the pylibs objects
        gd = grid.pylibs_hgrid
        vd = grid.pylibs_vgrid

        # Get the number of elements, nodes, and sides
        ne = gd.ne
        np_grid = gd.np

        # For ns (number of sides), we need to check if it's available
        # If not, we can use the length of the side arrays if available
        if hasattr(gd, "ns"):
            ns = gd.ns
        elif hasattr(gd, "isidenode"):
            ns = len(gd.isidenode)
        else:
            # If we can't determine ns, use a reasonable default
            # For triangular meshes, ns is approximately 1.5 * ne
            ns = int(ne * 1.5)
            logger.warning(
                f"Could not determine number of sides, using estimated value: {ns}"
            )

        # Get the number of vertical layers
        nvrt = vd.nvrt

        # Get node coordinates
        lxi = gd.x % 360  # Convert to 0-360 longitude range
        lyi = gd.y
        lzi0 = np.abs(vd.compute_zcor(gd.dp)).T  # Depth of each level, positive down

        # Source coordinates, in increasing order for the interpolation indices
        ds = ds.sortby([self.coords.x, self.coords.y, self.coords.z])
        sx = np.array(ds[self.coords.x].values) % 360
        sy = np.array(ds[self.coords.y].values)
        sz = np.abs(np.array(ds[self.coords.z].values))

        # Levels above or below the source's depths take its nearest depth
        lzi0 = np.clip(lzi0, sz.min(), sz.max())

        # Horizontal interpolation indices and weights, the same for every level
        idx = np.clip(((lxi[:, None] - sx[None, :]) >= 0).sum(axis=1) - 1, 0, len(sx) - 2)
        ratx = (lxi - sx[idx]) / (sx[idx + 1] - sx[idx])
        idy = np.clip(((lyi[:, None] - sy[None, :]) >= 0).sum(axis=1) - 1, 0, len(sy) - 2)
        raty = (lyi - sy[idy]) / (sy[idy + 1] - sy[idy])

        tracers = []
        for svar in [self.temp_var, self.salt_var]:
            if svar not in ds.variables:
                raise ValueError(f"Variable {svar} not found in the hotstart source")
            source = ds[svar].transpose(self.coords.z, self.coords.y, self.coords.x)
            cv = fill_land(fill_below_seabed(source.values, axis=0))

            levels = []
            for k in range(nvrt):
                logger.debug(f"Interpolating {svar} at level {k+1}/{nvrt}")
                lzi = lzi0[k]
                idz = np.clip(((lzi[:, None] - sz[None, :]) >= 0).sum(axis=1) - 1, 0, len(sz) - 2)
                ratz = (lzi - sz[idz]) / (sz[idz + 1] - sz[idz])

                # Trilinear interpolation
                v11 = cv[idz, idy, idx] * (1 - ratx) + cv[idz, idy, idx + 1] * ratx
                v12 = cv[idz, idy + 1, idx] * (1 - ratx) + cv[idz, idy + 1, idx + 1] * ratx
                v1 = v11 * (1 - raty) + v12 * raty
                v21 = cv[idz + 1, idy, idx] * (1 - ratx) + cv[idz + 1, idy, idx + 1] * ratx
                v22 = (
                    cv[idz + 1, idy + 1, idx] * (1 - ratx)
                    + cv[idz + 1, idy + 1, idx + 1] * ratx
                )
                v2 = v21 * (1 - raty) + v22 * raty
                levels.append(v1 * (1 - ratz) + v2 * ratz)
            tracers.append(np.array(levels))

        # Tracers at nodes, (node, level, tracer)
        tr_nd = np.stack(tracers).T

        # Calculate element tracers from node tracers
        try:
            tr_el = tr_nd[gd.elnode[:, :3]].mean(axis=1)
        except Exception as e:
            logger.error(f"Error calculating element tracers: {e}")
            # Create a fallback array with zeros
            tr_el = np.zeros((ne, nvrt, 2))
            logger.warning("Using zeros as fallback for element tracers")

        # Create NetCDF structure
        nd = zdata()
        nd.file_format = "NETCDF4"
        nd.dimname = ["node", "elem", "side", "nVert", "ntracers", "one"]
        nd.dims = [np_grid, ne, ns, nvrt, 2, 1]

        # Define variables
        nd.vars = [
            "time",
            "iths",
            "ifile",
            "idry_e",
            "idry_s",
            "idry",
            "eta2",
            "we",
            "tr_el",
            "tr_nd",
            "tr_nd0",
            "su2",
            "sv2",
            "q2",
            "xl",
            "dfv",
            "dfh",
            "dfq1",
            "dfq2",
            "nsteps_from_cold",
            "cumsum_eta",
        ]

        # Initialize variables
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(0.0)
        nd.time = vi
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(0).astype("int")
        nd.iths = vi
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(1).astype("int")
        nd.ifile = vi
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(0).astype("int")
        nd.nsteps_from_cold = vi

        vi = zdata()
        vi.dimname = ("elem",)
        vi.val = np.zeros(ne).astype("int32")
        nd.idry_e = vi
        vi = zdata()
        vi.dimname = ("side",)
        vi.val = np.zeros(ns).astype("int32")
        nd.idry_s = vi
        vi = zdata()
        vi.dimname = ("node",)
        vi.val = np.zeros(np_grid).astype("int32")
        nd.idry = vi
        vi = zdata()
        vi.dimname = ("node",)
        vi.val = np.zeros(np_grid)
        nd.eta2 = vi
        vi = zdata()
        vi.dimname = ("node",)
        vi.val = np.zeros(np_grid)
        nd.cumsum_eta = vi

        vi = zdata()
        vi.dimname = ("elem", "nVert")
        vi.val = np.zeros([ne, nvrt])
        nd.we = vi
        vi = zdata()
        vi.dimname = ("side", "nVert")
        vi.val = np.zeros([ns, nvrt])
        nd.su2 = vi
        vi = zdata()
        vi.dimname = ("side", "nVert")
        vi.val = np.zeros([ns, nvrt])
        nd.sv2 = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.q2 = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.xl = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfv = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfh = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfq1 = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfq2 = vi

        vi = zdata()
        vi.dimname = ("elem", "nVert", "ntracers")
        vi.val = tr_el
        nd.tr_el = vi
        vi = zdata()
        vi.dimname = ("node", "nVert", "ntracers")
        vi.val = tr_nd
        nd.tr_nd = vi
        vi = zdata()
        vi.dimname = ("node", "nVert", "ntracers")
        vi.val = tr_nd
        nd.tr_nd0 = vi

        # Write NetCDF file
        WriteNC(str(output_path), nd)
        logger.debug(f"Created hotstart file: {output_path}")

        return str(output_path)
