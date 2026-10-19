"""pyTMD 3 tidal BC: constituents import, overlay JSON, interpolate path."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pyTMD")
pytest.importorskip("rompy_schism")

import pyTMD.constituents
from pyTMD.io.model import load_database

_OVERLAY = Path(__file__).resolve().parents[1] / "test_data" / "tides" / "database.json"


class _LocalAmpPhase:
    """Minimal mapping so ``name in ds`` / ``ds[name].tmd`` work."""

    def __init__(self, names, npts):
        self.data_vars = list(names)
        self._data = {
            name: SimpleNamespace(
                tmd=SimpleNamespace(
                    amplitude=np.ones(npts),
                    phase=np.zeros(npts),
                )
            )
            for name in names
        }

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]


def test_bctides_imports_constituents():
    """bctides uses pyTMD.constituents (pyTMD 3); no arguments fallback."""
    from rompy_schism import bctides

    assert bctides._tmd_args is pyTMD.constituents
    for name in (
        "nodal_modulation",
        "frequency",
        "coefficients_table",
        "_constituent_parameters",
    ):
        assert hasattr(bctides._tmd_args, name), name
    assert not hasattr(bctides.Bctides, "_interpolate_tidal_data_v2")


def test_fixture_overlay_lists_oceanum_atlas():
    """Test extra_databases JSON is pyTMD 3 (model-name keys, z/u/v groups)."""
    params = load_database(extra_databases=[_OVERLAY])
    assert "OCEANUM-atlas" in list(params.keys())
    model = params["OCEANUM-atlas"]
    for group in ("z", "u", "v"):
        assert group in model, group


def test_get_tidal_factors_m2_fes():
    """Nodal/frequency call sites run on pyTMD.constituents."""
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tnames = ["m2"]
    bc.tidal_model = "FES2014"
    bc._start_time = datetime(2020, 1, 1)
    bc.amp = []
    bc._get_tidal_factors()
    assert len(bc.freq) == 1
    assert bc.freq[0] > 0
    assert bc.nodal_factor[0] > 0


def test_interp_group_opens_chunked_then_crops():
    """Regional path: chunks at open, then tmd.crop(bounds)."""
    from rompy_schism import bctides
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tide_interpolation_method = "linear"
    bc.extrapolate_tides = False
    bc.extrapolation_distance = 0.0

    cons = ["M2"]
    lons = np.array([150.0, 151.0])
    lats = np.array([-23.0, -24.0])
    npts = lons.size

    local = _LocalAmpPhase(cons, npts)
    cropped = MagicMock()
    cropped.tmd.constituents = cons
    cropped.tmd.coords_as.return_value = (lons, lats)
    cropped.tmd.interp.return_value = local
    cropped.__getitem__.side_effect = lambda _names: cropped

    opened = MagicMock()
    opened.tmd.crs.is_geographic = True
    opened.tmd.crop.return_value = cropped

    model = MagicMock()
    model.open_dataset.return_value = opened

    amp, pha = bc._interp_group(model, "z", lons, lats, cons)

    model.reduce_constituents.assert_called_once_with(cons, group="z")
    model.open_dataset.assert_called_once_with(
        group="z",
        constituents=cons,
        chunks=bctides._OPEN_CHUNKS,
    )
    opened.tmd.crop.assert_called_once_with(bc._bounds_for_points(lons, lats), buffer=0)
    cropped.tmd.to_units.assert_not_called()
    np.testing.assert_array_equal(amp, np.ones((2, 1)))
    np.testing.assert_array_equal(pha, np.zeros((2, 1)))


def test_interp_group_converts_currents_with_pytmd_units():
    """u/v amplitudes are scaled by Dataset.tmd.to_units, not a later /100."""
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tide_interpolation_method = "linear"
    bc.extrapolate_tides = False
    bc.extrapolation_distance = 0.0

    cons = ["m2"]
    lons = np.array([150.0, 151.0])
    lats = np.array([-23.0, -24.0])
    # Amplitudes stand in for values already converted to m/s.
    local = _LocalAmpPhase(cons, lons.size)

    converted = MagicMock()
    converted.data_vars = {}
    converted.tmd.constituents = cons
    converted.tmd.coords_as.return_value = (lons, lats)
    converted.tmd.interp.return_value = local
    converted.__getitem__.side_effect = lambda _names: converted

    cropped = MagicMock()
    cropped.tmd.to_units.return_value = converted

    opened = MagicMock()
    opened.tmd.crs.is_geographic = True
    opened.tmd.crop.return_value = cropped

    model = MagicMock()
    model.open_dataset.return_value = opened

    amp, pha = bc._interp_group(model, "u", lons, lats, cons)

    cropped.tmd.to_units.assert_called_once_with("m/s")
    converted.tmd.interp.assert_called_once()
    cropped.tmd.inpaint.assert_not_called()
    np.testing.assert_array_equal(amp, np.ones((2, 1)))
    np.testing.assert_array_equal(pha, np.zeros((2, 1)))


@pytest.mark.parametrize("n_iter", [0, 4])
def test_bilinear_fills_masked_cells_before_linear(n_iter):
    """Legacy bilinear inpaints masked cells, then uses xarray linear."""
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tide_interpolation_method = "bilinear"
    bc.tide_inpaint_iterations = n_iter
    bc.extrapolate_tides = False
    bc.extrapolation_distance = 50.0

    cons = ["m2"]
    lons = np.array([152.2])
    lats = np.array([-24.5])
    local = _LocalAmpPhase(cons, lons.size)
    filled = MagicMock()
    filled.tmd.coords_as.return_value = (lons, lats)
    filled.tmd.interp.return_value = local

    cropped = MagicMock()
    cropped.chunks = {}
    cropped.tmd.constituents = cons
    cropped.tmd.crs.is_geographic = True
    cropped.tmd.inpaint.return_value = filled
    cropped.__getitem__.side_effect = lambda _names: cropped

    opened = MagicMock()
    opened.tmd.crs.is_geographic = True
    opened.tmd.crop.return_value = cropped

    model = MagicMock()
    model.open_dataset.return_value = opened

    amp, pha = bc._interp_group(model, "z", lons, lats, cons)

    cropped.tmd.inpaint.assert_called_once_with(
        N=bc.tide_inpaint_iterations, is_geographic=True
    )
    filled.tmd.interp.assert_called_once()
    assert filled.tmd.interp.call_args.kwargs["method"] == "linear"
    assert filled.tmd.interp.call_args.kwargs["extrapolate"] is False
    np.testing.assert_array_equal(amp, np.ones((1, 1)))
    np.testing.assert_array_equal(pha, np.zeros((1, 1)))


def test_tide_inpaint_iterations_defaults_to_nearest_neighbor():
    """N=0 is the pyTMD inpaint default and the TidalDataset/Bctides default."""
    from rompy_schism.bctides import Bctides
    from rompy_schism.boundary_core import TidalDataset

    dataset = TidalDataset(tide_interpolation_method="Bilinear")
    assert dataset.tide_interpolation_method == "bilinear"
    assert dataset.tide_inpaint_iterations == 0

    bc = Bctides(hgrid=None, constituents=["m2"])
    assert bc.tide_inpaint_iterations == 0


def test_tide_interpolation_method_rejects_unknown_names():
    """Supported set is bilinear, linear, and nearest."""
    from pydantic import ValidationError

    from rompy_schism.boundary_core import TidalDataset

    with pytest.raises(ValidationError):
        TidalDataset(tide_interpolation_method="cubic")


def test_spline_is_deprecated_alias_of_bilinear():
    """spline warns and is stored as the coastal bilinear path."""
    from rompy_schism.bctides import Bctides
    from rompy_schism.boundary_core import TidalDataset

    with pytest.warns(DeprecationWarning, match="spline"):
        dataset = TidalDataset(tide_interpolation_method="SPLINE")
    assert dataset.tide_interpolation_method == "bilinear"

    with pytest.warns(DeprecationWarning, match="spline"):
        bc = Bctides(
            hgrid=None,
            constituents=["m2"],
            tide_interpolation_method="spline",
        )
    assert bc.tide_interpolation_method == "bilinear"
    assert bc._xarray_interp_method() == ("linear", True)


def test_bctides_rejects_negative_inpaint_iterations():
    from rompy_schism.bctides import Bctides

    with pytest.raises(ValueError, match="tide_inpaint_iterations"):
        Bctides(hgrid=None, constituents=["m2"], tide_inpaint_iterations=-1)


@pytest.mark.parametrize("data_type", ["h", "uv"])
def test_oceanum_coastal_interpolation_is_finite(tidal_data_files, data_type):
    """Wet coastal nodes must not silently produce non-finite harmonics."""
    from rompy_schism.bctides import Bctides

    # Grid node 2133. xarray linear is NaN here: elevation has a masked
    # corner, and the u stencil is entirely masked (nearest wet u ~52 km).
    # bilinear nearest-fills those cells before linear interpolation.
    bc = Bctides(
        hgrid=None,
        constituents=["m2"],
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
        tide_interpolation_method="bilinear",
        extrapolate_tides=False,
        extrapolation_distance=50.0,
        extra_databases=[tidal_data_files / "database.json"],
    )

    result = bc._interpolate_tidal_data(
        np.array([152.2166853158]),
        np.array([-24.4772773961]),
        ["m2"],
        data_type,
    )

    assert np.isfinite(result).all(), (
        f"{data_type} interpolation returned non-finite tidal coefficients: {result!r}"
    )


def test_elevation_from_database_z_only():
    """Elevation-only extracts must not require u/v model files."""
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tidal_database = None
    bc.extra_databases = []
    bc.tidal_model = "FES2014"
    bc.tide_interpolation_method = "linear"
    bc.extrapolate_tides = False
    bc.extrapolation_distance = 0.0

    tmd_model = MagicMock()
    model = MagicMock()
    tmd_model.from_database.return_value = model
    amp = np.ones((2, 1))
    pha = np.zeros((2, 1))
    bc._interp_group = MagicMock(return_value=(amp, pha))

    with patch("rompy_schism.bctides.pyTMD.io.model", return_value=tmd_model):
        out = bc._interpolate_tidal_data(
            np.array([150.0, 151.0]),
            np.array([-23.0, -24.0]),
            ["m2"],
            "h",
        )

    tmd_model.from_database.assert_called_once_with("FES2014", group=("z",))
    assert out.shape == (2, 1, 2)


def test_uv_from_database_zuv():
    """Velocity extracts still load z+u+v metadata."""
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tidal_database = None
    bc.extra_databases = []
    bc.tidal_model = "FES2014"
    bc.tide_interpolation_method = "linear"
    bc.extrapolate_tides = False
    bc.extrapolation_distance = 0.0

    tmd_model = MagicMock()
    model = MagicMock()
    tmd_model.from_database.return_value = model
    amp = np.ones((2, 1))
    pha = np.zeros((2, 1))
    bc._interp_group = MagicMock(return_value=(amp, pha))

    with patch("rompy_schism.bctides.pyTMD.io.model", return_value=tmd_model):
        out = bc._interpolate_tidal_data(
            np.array([150.0, 151.0]),
            np.array([-23.0, -24.0]),
            ["m2"],
            "uv",
        )

    tmd_model.from_database.assert_called_once_with("FES2014", group=("z", "u", "v"))
    assert out.shape == (2, 1, 4)
    # Conversion lives in _interp_group; the caller must not scale again.
    np.testing.assert_array_equal(out[:, :, 0], amp)
    np.testing.assert_array_equal(out[:, :, 2], amp)


def test_oceanum_currents_divide_transport_after_interpolation(tidal_data_files):
    """u/v follow OTPS/TMD/pyTMD 2: interp(transport) / interp(depth).

    pyTMD 3 opens OTIS/ATLAS currents as transport / depth on the model
    grid. Interpolating that ratio differs from interpolating transport
    and depth separately by up to ~2.4x in amplitude on the test boundary
    (node 9 for u, node 64 for v below). Reference values are built from
    the raw transport (group ``U``/``V``) and u/v-node bathymetry with
    plain xarray linear interpolation.
    """
    import xarray as xr
    from pyTMD.io import ATLAS

    from rompy_schism.bctides import Bctides

    lons = np.array([153.158071151, 148.9044003348, 150.857426192])
    lats = np.array([-23.6490629763, -18.7350010445, -19.6865789161])
    atlas = tidal_data_files / "oceanum-atlas"
    grid = atlas / "grid_tpxo9_atlas_30_v2.nc"
    files = [atlas / "u_m2_tpxo9_atlas_30_v2.nc"]

    bc = Bctides(
        hgrid=None,
        constituents=["m2"],
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
        tide_interpolation_method="linear",
        extra_databases=[tidal_data_files / "database.json"],
    )
    result = bc._interpolate_tidal_data(lons, lats, ["m2"], "uv")

    X = xr.DataArray(lons, dims="p")
    Y = xr.DataArray(lats, dims="p")
    for group, col in (("u", 0), ("v", 2)):
        depth = ATLAS.open_atlas_grid(grid, group=group)["bathymetry"]
        transport = ATLAS.open_mfdataset(files, group=group.upper())["m2"]
        ref = (
            transport.interp(x=X, y=Y, method="linear")
            / depth.interp(x=X, y=Y, method="linear")
        ).values
        # fixture labels: transport cm^2/s, depth m -> 1e-4 to m/s
        np.testing.assert_allclose(result[:, 0, col], np.abs(ref) * 1e-4, rtol=1e-5)
        ref_phase = np.degrees(np.arctan2(-ref.imag, ref.real)) % 360.0
        dphase = (result[:, 0, col + 1] - ref_phase + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(dphase, 0.0, atol=1e-4)


def _bounds(lons, lats, distance=0.0):
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.extrapolation_distance = distance
    return bc._bounds_for_points(np.asarray(lons, dtype=float), np.asarray(lats))


def test_crop_bounds_keep_a_compact_western_box():
    """Negative longitudes stay a short box; pyTMD main shifts them onto 0–360."""
    # 0.5° latitude pad; longitude pad is wider by 1/cos(20°).
    bounds = _bounds([-75.0, -70.0], [10.0, 20.0])
    lon_pad = 0.5 / np.cos(np.deg2rad(20.0))
    assert bounds[0] < bounds[1]
    assert bounds == pytest.approx([-75.0 - lon_pad, -70.0 + lon_pad, 9.5, 20.5])


def test_crop_bounds_cross_the_dateline_on_the_short_arc():
    """xmin > xmax is the pyTMD#613 wrap window, not min/max of ±180."""
    bounds = _bounds([179.0, -179.0], [-20.0, -19.0])
    lon_pad = 0.5 / np.cos(np.deg2rad(20.0))
    assert bounds[0] > bounds[1]
    assert bounds == pytest.approx([179.0 - lon_pad, -179.0 + lon_pad, -20.5, -18.5])


def test_unset_tide_method_uses_bilinear_fill():
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tide_interpolation_method = None
    assert bc._xarray_interp_method() == ("linear", True)


def test_unknown_tide_method_warns_and_skips_fill(caplog):
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tide_interpolation_method = "cubic"
    with caplog.at_level("WARNING"):
        assert bc._xarray_interp_method() == ("linear", False)
    assert "cubic" in caplog.text


def test_inpaint_iterations_reject_fractions_and_bools():
    from rompy_schism.bctides import Bctides
    from rompy_schism.boundary_core import TidalDataset

    for value in (1.5, True):
        with pytest.raises(ValueError, match="integer"):
            Bctides(hgrid=None, constituents=["m2"], tide_inpaint_iterations=value)
        with pytest.raises(ValueError, match="integer"):
            TidalDataset(tide_inpaint_iterations=value)


def test_currents_without_bathymetry_are_left_alone():
    """FES stores currents, so the transport workaround does not apply."""
    import xarray as xr

    from rompy_schism.bctides import Bctides

    ds = xr.Dataset({"m2": (("y", "x"), np.ones((2, 2)))})
    out, transport = Bctides._currents_to_transport(ds)
    assert transport is False
    assert out is ds


def test_nonpositive_bathymetry_becomes_missing():
    """OTIS-binary and TMD3 store land depth as 0. That must not stay finite."""
    import xarray as xr

    from rompy_schism.bctides import Bctides

    ds = xr.Dataset(
        {
            "m2": (("y", "x"), np.array([[1.0 + 0.0j, np.nan], [2.0 + 0.0j, 3.0]])),
            "bathymetry": (("y", "x"), np.array([[10.0, 0.0], [-1.0, 20.0]])),
            "mask": (("y", "x"), np.ones((2, 2), dtype=np.int8)),
        }
    )
    out, transport = Bctides._currents_to_transport(ds)
    assert transport is True
    assert np.isnan(out["bathymetry"].values[0, 1])
    assert np.isnan(out["bathymetry"].values[1, 0])
    fields = Bctides._fields_for_interpolation(out, transport=True)
    assert list(fields.data_vars) == ["m2", "bathymetry"]


def test_inpaint_keeps_wet_cells_for_nearest_and_dct():
    """N=0 and N>0 both put the original finite samples back."""
    from pyTMD.interpolate import inpaint

    xs = np.linspace(150.0, 151.0, 3)
    ys = np.linspace(-25.0, -24.0, 3)
    values = np.array(
        [[1.0, np.nan, 1.5], [np.nan, 2.0, 2.5], [3.0, 3.5, np.nan]],
        dtype=float,
    )
    wet = np.isfinite(values)
    for iterations in (0, 4):
        filled = inpaint(xs, ys, values, N=iterations, is_geographic=True)
        assert np.isfinite(filled).all()
        np.testing.assert_allclose(filled[wet], values[wet])


def test_tidal_dataset_get_copies_databases_and_extrapolation():
    from rompy_schism.boundary_core import TidalDataset

    dataset = TidalDataset(
        constituents=["m2"],
        extra_databases=[],
        extrapolate_tides=True,
        extrapolation_distance=80.0,
    )
    got = dataset.get(None)
    got["extra_databases"].append("mutated")
    assert dataset.extra_databases == []
    assert got["extrapolate_tides"] is True
    assert got["extrapolation_distance"] == 80.0


def test_finite_check_names_the_method_for_strict_linear():
    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.tide_interpolation_method = "linear"
    bc.extrapolate_tides = False
    with pytest.raises(ValueError, match="bilinear") as exc:
        bc._require_finite_harmonics(np.array([[[np.nan, 0.0]]]), "h")
    assert "method='linear'" in str(exc.value)


def test_projected_crop_bounds_are_metres():
    import xarray as xr

    from rompy_schism.bctides import Bctides

    bc = Bctides.__new__(Bctides)
    bc.extrapolation_distance = 50.0
    ds = xr.Dataset(
        {"m2": (("y", "x"), np.ones((2, 2)))},
        coords={"x": [400000.0, 401000.0], "y": [7200000.0, 7201000.0]},
        attrs={"crs": "EPSG:32756"},
    )
    bounds = bc._crop_bounds(ds, [151.0], [-24.0])
    assert bounds[1] - bounds[0] < 500_000
    assert min(bounds[0], bounds[1]) > 10_000
