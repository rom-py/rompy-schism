"""Tests for interpolating 3D ocean data to SCHISM's boundary nodes and levels."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rompy.core.data import DataBlob
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords

from rompy_schism.data import SCHISMDataBoundary
from rompy_schism.fill import fill_below_seabed, fill_land
from rompy_schism.grid import SCHISMGrid
from rompy_schism.hotstart import SCHISMDataHotstart

HERE = Path(__file__).parent.parent
PERIOD = TimeRange(start="2023-01-02", end="2023-01-03", interval="1h")
COORDS = DatasetCoords(t="time", x="lon", y="lat", z="depth")
DEPTH = np.array([0.5, 10, 50, 100, 300, 600, 1000, 2000, 3000])


def temperature(depth):
    """A profile that linear interpolation reproduces between the source depths."""
    return np.interp(depth, DEPTH, 25 - DEPTH / 150)


@pytest.fixture(scope="module")
def grid():
    return SCHISMGrid(
        hgrid=DataBlob(source=HERE / "test_data/hgrid.gr3"),
        vgrid=DataBlob(source=HERE / "test_data/vgrid.in"),
        drag=1,
    )


def ocean(path, seabed=None):
    """An ocean dataset file over the test grid, missing below `seabed` if given.

    The second day is 1 degree warmer than the first.
    """
    lon = np.arange(144.5, 155.5, 0.5)
    lat = np.arange(-25.5, -15.5, 0.5)
    profile = temperature(DEPTH)
    values = np.broadcast_to(
        profile[None, :, None, None], (2, len(DEPTH), len(lat), len(lon))
    ).copy()
    values[1] += 1
    if seabed is not None:
        values[:, DEPTH > seabed, :, ::2] = np.nan
    ds = xr.Dataset(
        {
            "temp": (("time", "depth", "lat", "lon"), values),
            "salt": (("time", "depth", "lat", "lon"), np.full(values.shape, 35.0)),
        },
        coords={
            "time": pd.date_range("2023-01-01", periods=2, freq="1D"),
            "depth": DEPTH,
            "lat": lat,
            "lon": lon,
        },
    )
    uri = path / f"ocean_{seabed}.nc"
    ds.to_netcdf(uri)
    return SourceFile(uri=uri)


def boundary_levels(grid):
    gd, vd = grid.pylibs_hgrid, grid.pylibs_vgrid
    nodes = np.concatenate([gd.iobn[i] for i in range(gd.nob)])
    return vd.compute_zcor(gd.dp[nodes])


class TestFill:
    def test_below_seabed_extends_the_deepest_value(self):
        values = np.array([[1.0, 2.0], [3.0, np.nan], [np.nan, np.nan]])
        filled = fill_below_seabed(values, axis=0)
        np.testing.assert_array_equal(filled, [[1, 2], [3, 2], [3, 2]])

    def test_below_seabed_along_another_axis(self):
        values = np.array([[1.0, 3.0, np.nan]])
        np.testing.assert_array_equal(fill_below_seabed(values, axis=1), [[1, 3, 3]])

    def test_land_takes_the_nearest_wet_column(self):
        values = np.array([[[1.0, np.nan, np.nan, 4.0]]])
        np.testing.assert_array_equal(fill_land(values), [[[1, 1, 4, 4]]])

    def test_land_everywhere_is_an_error(self):
        with pytest.raises(ValueError, match="no valid values"):
            fill_land(np.full((2, 2, 2), np.nan))


class TestBoundary:
    def get(self, grid, tmp_path, seabed=None):
        boundary = SCHISMDataBoundary(
            id="TEM_3D",
            source=ocean(tmp_path, seabed),
            variables=["temp"],
            coords=COORDS,
        )
        return xr.open_dataset(boundary.get(tmp_path, grid, PERIOD), decode_times=False)

    def test_profiles_at_the_vertical_grid_levels(self, grid, tmp_path):
        ds = self.get(grid, tmp_path)
        zcor = boundary_levels(grid)
        np.testing.assert_allclose(ds.vertical_levels, zcor)
        # The first record is the second day, the start of the period
        expected = temperature(np.clip(-zcor, DEPTH[0], None)) + 1
        np.testing.assert_allclose(ds.time_series[0, :, :, 0], expected, atol=1e-6)

    def test_source_seabed_does_not_cut_the_profiles(self, grid, tmp_path):
        """Where some source columns around a node stop at 300 m, the profile below
        still takes the colder water of the deeper columns."""
        ds = self.get(grid, tmp_path, seabed=300)
        values = ds.time_series[0, :, :, 0].values
        assert not np.isnan(values).any()
        assert (np.diff(values, axis=1) >= -1e-9).all()
        zcor = boundary_levels(grid)
        deepest = np.argmin(zcor[:, 0])
        at_300m = np.interp(300, -zcor[deepest][::-1], values[deepest][::-1])
        assert values[deepest, 0] < at_300m - 1


class TestHotstart:
    def get(self, grid, tmp_path, seabed=None):
        hotstart = SCHISMDataHotstart(
            source=ocean(tmp_path, seabed),
            coords=COORDS,
            temp_var="temp",
            salt_var="salt",
        )
        return xr.open_dataset(hotstart.get(tmp_path, grid, PERIOD))

    def test_takes_the_time_closest_to_the_start(self, grid, tmp_path):
        ds = self.get(grid, tmp_path)
        gd, vd = grid.pylibs_hgrid, grid.pylibs_vgrid
        depth = np.clip(np.abs(vd.compute_zcor(gd.dp)), DEPTH[0], None)
        np.testing.assert_allclose(ds.tr_nd[:, :, 0], temperature(depth) + 1, atol=1e-6)

    def test_source_seabed_keeps_profiles_monotonic(self, grid, tmp_path):
        ds = self.get(grid, tmp_path, seabed=300)
        values = ds.tr_nd[:, :, 0].values
        assert not np.isnan(values).any()
        assert (np.diff(values, axis=1) >= -1e-9).all()
        assert values.min() > temperature(3000)
