import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rompy.core import BaseGrid, DataBlob, DataGrid, TimeRange
from rompy.core.data import (SourceDatamesh, SourceDataset, SourceFile,
                             SourceIntake)
from rompy.schism.data import SCHISMDataSflux, SfluxAir
from rompy.schism.namelists import Sflux_Inputs

HERE = Path(__file__).parent
DATAMESH_TOKEN = os.environ.get("DATAMESH_TOKEN")


@pytest.fixture
def grid_data_source():
    return SourceIntake(
        dataset_id="era5",
        catalog_uri=HERE / ".." / "data" / "catalog.yaml",
    )


def test_atmos(tmp_path, grid_data_source):
    data = SCHISMDataSflux(
        air_1=SfluxAir(
            id="air_1",
            source=grid_data_source,
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
