import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rompy.core import BaseGrid, DataBlob, DataGrid, TimeRange
from rompy.core.data import (SourceDatamesh, SourceDataset, SourceFile,
                             SourceIntake)
from rompy.schism import SCHISMGrid
from rompy.schism.data import (SCHISMDataBoundary, SCHISMDataOcean,
                               SCHISMDataSflux, SCHISMDataTides, SfluxAir,
                               TidalDataset)
from rompy.schism.namelists import Sflux_Inputs

HERE = Path(__file__).parent
DATAMESH_TOKEN = os.environ.get("DATAMESH_TOKEN")


@pytest.fixture
def grid2d():
    return SCHISMGrid(hgrid=DataBlob(source="test_data/hgrid.gr3"))


@pytest.fixture
def grid_atmos_source():
    return SourceIntake(
        dataset_id="era5",
        catalog_uri=HERE / ".." / "data" / "catalog.yaml",
    )


@pytest.fixture
def hycom_bnd():
    return SCHISMDataBoundary(
        id="hycom",
        source=SourceFile(
            uri=HERE / ".." / "data" / "hycom.nc",
        ),
        variable="surf_el",
        coords={"t": "time", "y": "ylat", "x": "xlon", "z": "depth"},
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


def test_oceandataboundary(tmp_path, grid2d, hycom_bnd):
    hycom_bnd.get(tmp_path, grid2d)


def test_oceandata(tmp_path, grid2d, hycom_bnd):
    oceandata = SCHISMDataOcean(elev2D=hycom_bnd)
    oceandata.get(tmp_path, grid2d)


def test_tidal_boundary(tmp_path, grid2d):
    tides = SCHISMDataTides(
        tidal_data=TidalDataset(
            elevations=Path(__file__).parent
            / ".."
            / "data"
            / "tides"
            / "h_m2s2n2k2k1o1p1q1mmmfm4mn4ms42n2s12q1j1l2m3mu2nu2oo1.nc",
            velocities=Path(__file__).parent
            / ".."
            / "data"
            / "tides"
            / "u_m2s2n2k2k1o1p1q1mmmfm4mn4ms42n2s12q1j1l2m3mu2nu2oo1.nc",
        )
    )
    tides.get(
        destdir=tmp_path,
        grid=grid2d,
        time=TimeRange(start="2023-01-01", end="2023-01-02", dt=3600),
    )
