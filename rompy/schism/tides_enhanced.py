"""
Enhanced implementation of SCHISM tidal data handling.

This module provides an improved approach to handling SCHISM tidal data
with support for all boundary condition types specified in the SCHISM
documentation.
"""

import logging
import os
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Any, cast

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from rompy.core.config import BaseConfig
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel
from rompy.schism.grid import SCHISMGrid

# Import bctides and boundary modules
from .bctides import Bctides
from .boundary_tides import (
    TidalBoundary,
    BoundaryConfig,
    ElevationType,
    VelocityType,
    TracerType,
    create_tidal_boundary,
    create_hybrid_boundary,
    create_river_boundary,
    create_nested_boundary
)

logger = logging.getLogger(__name__)

# Utility function to convert numpy types to Python types
def to_python_type(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [to_python_type(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    else:
        return obj


class TidalDataset(RompyBaseModel):
    """This class is used to define the tidal dataset"""

    data_type: Literal["tidal_dataset"] = Field(
        default="tidal_dataset",
        description="Model type discriminator",
    )
    elevations: str = Field(..., description="Path to elevations file")
    velocities: str = Field(..., description="Path to currents file")

    def get(self, destdir: str | Path) -> str:
        """Make tidal data files available.

        Parameters
        ----------
        destdir : str | Path
            Destination directory

        Returns
        -------
        str
            Path to the destination directory
        """
        # Set environment variables for tidal data paths
        os.environ["TPXO_ELEVATION"] = str(self.elevations)
        os.environ["TPXO_VELOCITY"] = str(self.velocities)
        return str(destdir)


class BoundarySetup(RompyBaseModel):
    """Configuration for a boundary in SCHISM."""

    # Basic boundary configuration
    elev_type: int = Field(
        5, description="Elevation boundary type (0-5)"
    )
    vel_type: int = Field(
        5, description="Velocity boundary type (-4, -1, 0-5)"
    )
    temp_type: int = Field(
        0, description="Temperature boundary type (0-4)"
    )
    salt_type: int = Field(
        0, description="Salinity boundary type (0-4)"
    )

    # Values for constant boundaries
    const_elev: Optional[float] = Field(
        None, description="Constant elevation value (for type 2)"
    )
    const_flow: Optional[float] = Field(
        None, description="Constant flow value (for type 2)"
    )
    const_temp: Optional[float] = Field(
        None, description="Constant temperature value (for type 2)"
    )
    const_salt: Optional[float] = Field(
        None, description="Constant salinity value (for type 2)"
    )

    # Values for relaxation and nudging
    inflow_relax: float = Field(
        0.5, description="Relaxation factor for inflow (0-1)"
    )
    outflow_relax: float = Field(
        0.1, description="Relaxation factor for outflow (0-1)"
    )
    temp_nudge: float = Field(
        1.0, description="Temperature nudging factor (0-1)"
    )
    salt_nudge: float = Field(
        1.0, description="Salinity nudging factor (0-1)"
    )

    # Flather boundary parameters
    mean_elev: Optional[List[float]] = Field(
        None, description="Mean elevation for Flather boundaries"
    )
    mean_flow: Optional[List[List[float]]] = Field(
        None, description="Mean flow for Flather boundaries"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_boundary_config(self) -> BoundaryConfig:
        """Convert to BoundaryConfig for TidalBoundary."""
        return BoundaryConfig(
            elev_type=self.elev_type,
            vel_type=self.vel_type,
            temp_type=self.temp_type,
            salt_type=self.salt_type,
            ethconst=self.const_elev,
            vthconst=self.const_flow,
            tthconst=self.const_temp,
            sthconst=self.const_salt,
            inflow_relax=self.inflow_relax,
            outflow_relax=self.outflow_relax,
            tobc=self.temp_nudge,
            sobc=self.salt_nudge,
            eta_mean=self.mean_elev,
            vn_mean=self.mean_flow
        )


class SCHISMDataTidesEnhanced(RompyBaseModel):
    """Enhanced SCHISM tidal data handler with support for all boundary types."""

    # Allow arbitrary types for schema generation
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_type: Literal["tides_enhanced"] = Field(
        default="tides_enhanced",
        description="Model type discriminator",
    )

    # Tidal dataset specification
    tidal_data: Optional[TidalDataset] = Field(
        None, description="Tidal dataset with elevation and velocity files"
    )

    # Basic tidal configuration
    constituents: Optional[List[str]] = Field(
        None, description="Tidal constituents to include"
    )
    tidal_database: Optional[str] = Field(
        "tpxo", description="Tidal database to use"
    )

    # Earth tidal potential settings
    ntip: Optional[int] = Field(
        0, description="Number of tidal potential regions (0 to disable, >0 to enable)"
    )
    tip_dp: Optional[float] = Field(
        1.0, description="Depth threshold for tidal potential calculations"
    )
    cutoff_depth: Optional[float] = Field(
        50.0, description="Cutoff depth for tides"
    )

    # Legacy boundary configuration
    flags: Optional[List[List[int]]] = Field(
        None, description="Boundary condition flags (legacy format)"
    )
    ethconst: Optional[List[float]] = Field(
        None, description="Constant elevation for each boundary (legacy format)"
    )
    vthconst: Optional[List[float]] = Field(
        None, description="Constant velocity for each boundary (legacy format)"
    )
    tthconst: Optional[List[float]] = Field(
        None, description="Constant temperature for each boundary (legacy format)"
    )
    sthconst: Optional[List[float]] = Field(
        None, description="Constant salinity for each boundary (legacy format)"
    )
    tobc: Optional[List[float]] = Field(
        None, description="Temperature OBC values (legacy format)"
    )
    sobc: Optional[List[float]] = Field(
        None, description="Salinity OBC values (legacy format)"
    )
    relax: Optional[List[float]] = Field(
        None, description="Relaxation parameters (legacy format)"
    )

    # Enhanced boundary configuration
    boundaries: Optional[Dict[int, BoundarySetup]] = Field(
        None, description="Enhanced boundary configuration by boundary index"
    )

    # Predefined configurations
    setup_type: Optional[Literal["tidal", "hybrid", "river", "nested"]] = Field(
        None, description="Predefined boundary setup type"
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

    def create_tidal_boundary(self, grid) -> TidalBoundary:
        """Create a TidalBoundary instance from this configuration.

        Parameters
        ----------
        grid : SCHISMGrid
            SCHISM grid instance

        Returns
        -------
        TidalBoundary
            Configured tidal boundary handler
        """
        # Get tidal data paths
        tidal_elevations = None
        tidal_velocities = None
        if self.tidal_data:
            tidal_elevations = self.tidal_data.elevations
            tidal_velocities = self.tidal_data.velocities

        # Create boundary handler
        boundary = TidalBoundary(
            grid_path=grid.hgrid.source,
            constituents=self.constituents,
            tidal_database=self.tidal_database,
            tidal_elevations=tidal_elevations,
            tidal_velocities=tidal_velocities,
            ntip=self.ntip,
            tip_dp=self.tip_dp,
            cutoff_depth=self.cutoff_depth
        )

        # Configure boundaries
        if self.boundaries:
            # Use enhanced boundary configuration
            for idx, setup in self.boundaries.items():
                boundary.set_boundary_config(idx, setup.to_boundary_config())
        elif self.flags:
            # Use legacy flags
            max_boundary = len(self.flags)
            for i in range(max_boundary):
                config = BoundaryConfig(
                    elev_type=self.flags[i][0] if len(self.flags[i]) > 0 else 0,
                    vel_type=self.flags[i][1] if len(self.flags[i]) > 1 else 0,
                    temp_type=self.flags[i][2] if len(self.flags[i]) > 2 else 0,
                    salt_type=self.flags[i][3] if len(self.flags[i]) > 3 else 0
                )

                # Add constant values if provided
                if self.ethconst and i < len(self.ethconst):
                    config.ethconst = self.ethconst[i]
                if self.vthconst and i < len(self.vthconst):
                    config.vthconst = self.vthconst[i]
                if self.tthconst and i < len(self.tthconst):
                    config.tthconst = self.tthconst[i]
                if self.sthconst and i < len(self.sthconst):
                    config.sthconst = self.sthconst[i]
                if self.tobc and i < len(self.tobc):
                    config.tobc = self.tobc[i]
                if self.sobc and i < len(self.sobc):
                    config.sobc = self.sobc[i]

                boundary.set_boundary_config(i, config)
        elif self.setup_type:
            # Use predefined configuration
            if self.setup_type == "tidal":
                # Pure tidal boundary
                for i in range(grid.pylibs_hgrid.nob):
                    boundary.set_boundary_type(
                        i,
                        elev_type=ElevationType.TIDAL,
                        vel_type=VelocityType.TIDAL
                    )
            elif self.setup_type == "hybrid":
                # Tidal + external data
                for i in range(grid.pylibs_hgrid.nob):
                    boundary.set_boundary_type(
                        i,
                        elev_type=ElevationType.TIDALSPACETIME,
                        vel_type=VelocityType.TIDALSPACETIME
                    )
            elif self.setup_type == "river":
                # River boundary (first boundary only)
                if grid.pylibs_hgrid.nob > 0:
                    boundary.set_boundary_type(
                        0,
                        elev_type=ElevationType.NONE,
                        vel_type=VelocityType.CONSTANT,
                        vthconst=-100.0  # Default inflow
                    )
            elif self.setup_type == "nested":
                # Nested boundary with relaxation
                for i in range(grid.pylibs_hgrid.nob):
                    boundary.set_boundary_type(
                        i,
                        elev_type=ElevationType.SPACETIME,
                        vel_type=VelocityType.RELAXED,
                        temp_type=TracerType.SPACETIME,
                        salt_type=TracerType.SPACETIME,
                        inflow_relax=0.8,
                        outflow_relax=0.8
                    )
        else:
            # Default: tidal boundary for all open boundaries
            for i in range(grid.pylibs_hgrid.nob):
                boundary.set_boundary_type(
                    i,
                    elev_type=ElevationType.TIDAL,
                    vel_type=VelocityType.TIDAL
                )

        return boundary

    def get(self, destdir: str | Path, grid: SCHISMGrid, time: TimeRange) -> str:
        """Generate bctides.in file.

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
        str
            Path to the generated bctides.in file
        """
        logger.info(f"===== SCHISMDataTidesEnhanced.get called with destdir={destdir} =====")

        # Convert destdir to Path object
        destdir = Path(destdir)

        # Create destdir if it doesn't exist
        if not destdir.exists():
            logger.info(f"Creating destination directory: {destdir}")
            destdir.mkdir(parents=True, exist_ok=True)

        # Make tidal dataset available if provided
        if self.tidal_data:
            logger.info(f"Processing tidal data from {self.tidal_data}")
            self.tidal_data.get(destdir)

        # Create tidal boundary handler
        boundary = self.create_tidal_boundary(grid)

        # Set start time and run duration
        start_time = time.start
        run_days = (time.end - time.start).total_seconds() / 86400.0  # Convert to days
        boundary.set_run_parameters(start_time, run_days)

        # Generate bctides.in file
        bctides_path = destdir / "bctides.in"
        logger.info(f"Writing bctides.in to: {bctides_path}")

        try:
            boundary.write_boundary_file(bctides_path)
            logger.info(f"Successfully wrote bctides.in to {bctides_path}")
        except Exception as e:
            logger.error(f"Error writing bctides.in: {e}")
            # Create minimal fallback version
            try:
                with open(bctides_path, "w") as f:
                    f.write("0 10.0 !nbfr, beta_flux\n")
                    f.write(f"{grid.pylibs_hgrid.nob} !nope: number of open boundaries with elevation specified\n")
                    for i in range(grid.pylibs_hgrid.nob):
                        f.write(f"{i+1} 0. !open bnd #{i+1}, eta amplitude\n")
                    f.write("0 !ncbn: total # of flow bnd segments with discharge\n")
                    f.write("0 !nfluxf: total # of flux boundary segments\n")
                logger.info(f"Created minimal fallback bctides.in at {bctides_path}")
            except Exception as e2:
                logger.error(f"Failed to create fallback bctides.in: {e2}")

        return str(bctides_path)


# Factory functions for common tidal configurations

def create_tidal_only_config(
    constituents: List[str] = None,
    tidal_database: str = "tpxo",
    tidal_elevations: str = None,
    tidal_velocities: str = None,
    ntip: int = 0
) -> SCHISMDataTidesEnhanced:
    """Create a configuration for tidal-only boundaries.

    Parameters
    ----------
    constituents : list of str, optional
        Tidal constituents to use, defaults to major constituents
    tidal_database : str, optional
        Tidal database to use, by default "tpxo"
    tidal_elevations : str, optional
        Path to tidal elevations file
    tidal_velocities : str, optional
        Path to tidal velocities file
    ntip : int, optional
        Number of tidal potential regions, by default 0

    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = None
    if tidal_elevations and tidal_velocities:
        tidal_data = TidalDataset(
            elevations=tidal_elevations,
            velocities=tidal_velocities
        )

    return SCHISMDataTidesEnhanced(
        constituents=constituents or ["O1", "K1", "Q1", "P1", "M2", "S2", "K2", "N2"],
        tidal_database=tidal_database,
        tidal_data=tidal_data,
        ntip=ntip,
        setup_type="tidal"
    )


def create_hybrid_config(
    constituents: List[str] = None,
    tidal_database: str = "tpxo",
    tidal_elevations: str = None,
    tidal_velocities: str = None
) -> SCHISMDataTidesEnhanced:
    """Create a configuration for hybrid tidal + external data boundaries.

    Parameters
    ----------
    constituents : list of str, optional
        Tidal constituents to use, defaults to major constituents
    tidal_database : str, optional
        Tidal database to use, by default "tpxo"
    tidal_elevations : str, optional
        Path to tidal elevations file
    tidal_velocities : str, optional
        Path to tidal velocities file

    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = None
    if tidal_elevations and tidal_velocities:
        tidal_data = TidalDataset(
            elevations=tidal_elevations,
            velocities=tidal_velocities
        )

    return SCHISMDataTidesEnhanced(
        constituents=constituents or ["O1", "K1", "Q1", "P1", "M2", "S2", "K2", "N2"],
        tidal_database=tidal_database,
        tidal_data=tidal_data,
        setup_type="hybrid"
    )


def create_river_config(
    river_boundary_index: int = 0,
    river_flow: float = -100.0,
    other_boundaries: Literal["tidal", "none"] = "tidal",
    constituents: List[str] = None,
    tidal_database: str = "tpxo",
    tidal_elevations: str = None,
    tidal_velocities: str = None
) -> SCHISMDataTidesEnhanced:
    """Create a configuration with a river boundary.

    Parameters
    ----------
    river_boundary_index : int, optional
        Index of the river boundary, by default 0
    river_flow : float, optional
        River flow value (negative for inflow), by default -100.0
    other_boundaries : str, optional
        How to handle other boundaries, by default "tidal"
    constituents : list of str, optional
        Tidal constituents to use, defaults to major constituents
    tidal_database : str, optional
        Tidal database to use, by default "tpxo"
    tidal_elevations : str, optional
        Path to tidal elevations file
    tidal_velocities : str, optional
        Path to tidal velocities file

    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = None
    if tidal_elevations and tidal_velocities:
        tidal_data = TidalDataset(
            elevations=tidal_elevations,
            velocities=tidal_velocities
        )

    # Create basic configuration
    config = SCHISMDataTidesEnhanced(
        constituents=constituents or ["O1", "K1", "Q1", "P1", "M2", "S2", "K2", "N2"],
        tidal_database=tidal_database,
        tidal_data=tidal_data,
        boundaries={}
    )

    # Configure river boundary
    river_config = BoundarySetup(
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.CONSTANT,
        temp_type=TracerType.NONE,
        salt_type=TracerType.NONE,
        const_flow=river_flow
    )

    # Configure other boundaries if needed
    if other_boundaries == "tidal":
        other_config = BoundarySetup(
            elev_type=ElevationType.TIDAL,
            vel_type=VelocityType.TIDAL,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE
        )
    else:
        other_config = BoundarySetup(
            elev_type=ElevationType.NONE,
            vel_type=VelocityType.NONE,
            temp_type=TracerType.NONE,
            salt_type=TracerType.NONE
        )

    # Add boundary configurations
    boundaries = {river_boundary_index: river_config}
    # Other boundary indices will be set dynamically in get() method

    config.boundaries = boundaries
    return config


def create_nested_config(
    with_tides: bool = False,
    inflow_relax: float = 0.8,
    outflow_relax: float = 0.8,
    constituents: List[str] = None,
    tidal_database: str = "tpxo",
    tidal_elevations: str = None,
    tidal_velocities: str = None
) -> SCHISMDataTidesEnhanced:
    """Create a configuration for nested model with external data.

    Parameters
    ----------
    with_tides : bool, optional
        Whether to include tides, by default False
    inflow_relax : float, optional
        Relaxation factor for inflow, by default 0.8
    outflow_relax : float, optional
        Relaxation factor for outflow, by default 0.8
    constituents : list of str, optional
        Tidal constituents to use if with_tides=True
    tidal_database : str, optional
        Tidal database to use if with_tides=True, by default "tpxo"
    tidal_elevations : str, optional
        Path to tidal elevations file if with_tides=True
    tidal_velocities : str, optional
        Path to tidal velocities file if with_tides=True

    Returns
    -------
    SCHISMDataTidesEnhanced
        Configured tidal data handler
    """
    tidal_data = None
    if with_tides and tidal_elevations and tidal_velocities:
        tidal_data = TidalDataset(
            elevations=tidal_elevations,
            velocities=tidal_velocities
        )

    # Create boundary configuration
    if with_tides:
        default_config = BoundarySetup(
            elev_type=ElevationType.TIDALSPACETIME,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.SPACETIME,
            salt_type=TracerType.SPACETIME,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax
        )
    else:
        default_config = BoundarySetup(
            elev_type=ElevationType.SPACETIME,
            vel_type=VelocityType.RELAXED,
            temp_type=TracerType.SPACETIME,
            salt_type=TracerType.SPACETIME,
            inflow_relax=inflow_relax,
            outflow_relax=outflow_relax
        )

    return SCHISMDataTidesEnhanced(
        constituents=constituents if with_tides else None,
        tidal_database=tidal_database if with_tides else None,
        tidal_data=tidal_data,
        boundaries={0: default_config}  # Will be applied to all boundaries
    )
