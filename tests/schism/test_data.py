import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rompy.core import BaseGrid, DataBlob, DataGrid, TimeRange
from rompy.core.data import SourceDatamesh, SourceDataset, SourceFile, SourceIntake
from rompy.schism import SCHISMGrid2D, SCHISMGrid3D
from rompy.schism.data import SCHISMDataBoundary, SCHISMDataSflux, SfluxAir
from rompy.schism.namelists import Sflux_Inputs

HERE = Path(__file__).parent
DATAMESH_TOKEN = os.environ.get("DATAMESH_TOKEN")


@pytest.fixture
def grid2d():
    return SCHISMGrid2D(hgrid=DataBlob(source="test_data/hgrid.gr3"))


@pytest.fixture
def grid_atmos_source():
    return SourceIntake(
        dataset_id="era5",
        catalog_uri=HERE / ".." / "data" / "catalog.yaml",
    )


def test_atmos(tmp_path, grid_atmos_source):
    data = SCHISMDataSflux(
        air_1=SfluxAir(
            id="air_1",
            source=grid_atmos_source,
            filter={
                "sort": {"coords": ["latitude"]},
                "crop": {
                    "time": slice("2023-01-01", "2023-01-02"),
                    "latitude": slice(0, 20),
                    "longitude": slice(0, 20),
                },
            },
        )
    )
    data.get(tmp_path)


def test_oceansource(tmp_path, grid=grid2d):
    data = SCHISMDataBoundary(
        id="hycom",
        source=SourceFile(
            id="ocean_1",
            uri=HERE / ".." / "data" / "hycom.nc",
        ),
        variable="surf_el",
        coords={"t": "time", "y": "ylat", "x": "xlon", "z": "depth"},
    )
    grid = SCHISMGrid2D(hgrid=DataBlob(source="test_data/hgrid.gr3"))
    data.get(tmp_path, grid)
