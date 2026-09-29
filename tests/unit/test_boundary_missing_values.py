"""Boundary values missing from the source come from the nearest valid data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords

from rompy_schism.data import SCHISMDataBoundary

PERIOD = TimeRange(start="2023-01-01T00", end="2023-01-02T00", interval="1h")


def sea_level(tmp_path, grid2d, missing: slice) -> SCHISMDataBoundary:
    """A sea level varying with latitude over the grid, missing in a latitude band."""
    lon = np.arange(140.0, 160.0, 0.5)
    lat = np.arange(-30.0, -10.0, 0.5)
    values = np.broadcast_to(lat[None, :, None] / 100, (3, lat.size, lon.size)).copy()
    values[:, (lat >= missing.start) & (lat <= missing.stop), :] = np.nan
    ds = xr.Dataset(
        {"zos": (("time", "lat", "lon"), values)},
        coords={"time": pd.date_range("2023-01-01", periods=3), "lat": lat, "lon": lon},
    )
    ds.to_netcdf(tmp_path / "source.nc")
    return SCHISMDataBoundary(
        id="elev2D",
        source=SourceFile(uri=tmp_path / "source.nc"),
        variables=["zos"],
        coords=DatasetCoords(t="time", x="lon", y="lat"),
    )


def test_missing_values_come_from_the_nearest_boundary_nodes(tmp_path, grid2d):
    boundary = sea_level(tmp_path, grid2d, missing=slice(-18.0, -10.0))
    elev2d = xr.open_dataset(boundary.get(tmp_path, grid2d, PERIOD))
    values = elev2d.time_series.values[0, :, 0, 0]
    assert not np.isnan(values).any()
    # The filled nodes take the value at the edge of the valid data (near 18.5°S),
    # not the median of the boundary
    lat = grid2d.pylibs_hgrid.y[grid2d.pylibs_hgrid.iobn[0]]
    filled = lat > -18.0
    assert filled.any()
    assert np.allclose(values[filled], values[~filled][np.argmax(lat[~filled])])


def test_boundary_without_valid_data_raises(tmp_path, grid2d):
    boundary = sea_level(tmp_path, grid2d, missing=slice(-40.0, 0.0))
    with pytest.raises(ValueError, match="No valid zos data at the open boundary"):
        boundary.get(tmp_path, grid2d, PERIOD)


def test_fill_nearest_extends_a_single_valid_value():
    from rompy_schism.data import _fill_nearest

    filled = _fill_nearest(np.array([np.nan, 1.0, np.nan, np.nan]))
    assert np.array_equal(filled, [1.0, 1.0, 1.0, 1.0])
    assert np.array_equal(_fill_nearest(np.array([2.0, np.nan, 5.0])), [2.0, 2.0, 5.0])
