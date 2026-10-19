"""
Tidal boundary conditions for SCHISM.

A direct implementation based on PyLibs scripts/gen_bctides.py with no fallbacks.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pyTMD
import pyTMD.constituents as _tmd_args
import timescale
import xarray as xr
from rompy.formatting import ARROW
from rompy.logging import get_logger
from scipy.spatial import KDTree

logger = get_logger(__name__)

# pyTMD 3 Dataset.tmd.interp on a regular grid accepts only these xarray methods.
# "bilinear" is the historical default: xarray linear after a masked-cell inpaint.
_TIDE_INTERP_METHODS = frozenset({"bilinear", "linear", "nearest"})
_SPLINE_INTERP_DEPRECATION = (
    "tide_interpolation_method 'spline' is deprecated and will be removed. "
    "pyTMD 3 has no spline interpolator; 'spline' uses the same coastal-fill "
    "linear path as 'bilinear'. Supported methods are 'bilinear' (default), "
    "'linear', and 'nearest'."
)


def normalize_tide_interpolation_method(method, *, stacklevel=2):
    """Lower-case a tide interpolation name and retire ``spline``.

    ``spline`` warns and becomes ``bilinear``. ``None`` is returned unchanged
    so callers can apply their own default.
    """
    if method is None:
        return None
    name = str(method).strip().lower()
    if name == "spline":
        warnings.warn(
            _SPLINE_INTERP_DEPRECATION,
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return "bilinear"
    return name


def require_inpaint_iterations(value):
    """Return ``value`` as an int, or raise if it is not a whole number ≥ 0."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("tide_inpaint_iterations must be an integer >= 0")
    if int(value) < 0:
        raise ValueError("tide_inpaint_iterations must be an integer >= 0")
    return int(value)


try:
    import dask  # noqa: F401

    _OPEN_CHUNKS = "auto"
except ImportError:  # pragma: no cover - xarray chunks need dask
    _OPEN_CHUNKS = None


class Bctides:
    """Direct implementation of SCHISM tidal boundary conditions using PyLibs.

    Based on scripts/gen_bctides.py
    """

    def __init__(
        self,
        hgrid,
        flags=None,
        constituents="major",
        tidal_database=None,
        tidal_model="FES2014",
        tidal_potential=True,
        cutoff_depth=50.0,
        nodal_corrections=True,
        tide_interpolation_method="bilinear",
        tide_inpaint_iterations=0,
        extrapolate_tides=False,
        extrapolation_distance=50.0,
        extra_databases=[],
        mdt=None,
        ethconst=None,
        vthconst=None,
        tthconst=None,
        sthconst=None,
        tobc=None,
        sobc=None,
        relax=None,  # For backward compatibility
        inflow_relax=None,
        outflow_relax=None,
        eta_mean=None,
        vn_mean=None,
        nvrt=2,
        elev_th_path=None,
        elev_st_path=None,
        flow_th_path=None,
        vel_st_path=None,
        temp_th_path=None,
        temp_3d_path=None,
        salt_th_path=None,
        salt_3d_path=None,
    ):
        """Initialize Bctides handler.

        Parameters
        ----------
        hgrid : Grid or str
            SCHISM horizontal grid
        flags : list of lists, optional
            Boundary condition flags
        constituents : str or list, optional
            Tidal constituents to use, by default "major"
        tidal_database : path, optional
            Path to pyTMD tidal database to use, by default None which uses the default
        tidal_model : str, optional
            Tidal model name (e.g., 'FES2014'), by default 'FES2014'
        tidal_potential : bool, optional
            Whether to apply tidal potential, by default True
        cutoff_depth : float, optional
            Cutoff depth for tidal potential, by default 50.0
        nodal_corrections : bool, optional
            Whether to apply nodal corrections, by default True
        tide_interpolation_method : str, optional
            ``'bilinear'`` (default), ``'linear'``, or ``'nearest'``.
            ``linear`` and ``nearest`` match pyTMD 3 ``Dataset.tmd.interp``.
            ``bilinear`` is the historical name for xarray ``linear`` after
            ``Dataset.tmd.inpaint``, so wet coastal nodes stay finite.
            ``spline`` is deprecated and mapped to ``bilinear``.
        tide_inpaint_iterations : int, optional
            ``N`` passed to ``Dataset.tmd.inpaint`` on the bilinear path.
            0 (default) is nearest-neighbor fill. ``N > 0`` is that many
            DCT penalized least-squares iterations. Ignored for ``linear``
            and ``nearest``. Must be >= 0.
        extrapolation_distance : float, optional
            Kilometres. Pads the tide-model crop (at least half a degree)
            and, when ``extrapolate_tides`` is True, limits pyTMD's
            out-of-domain fill. Default 50, matching ``TidalDataset``.
        ethconst : list, optional
            Constant elevation for each boundary
        vthconst : list, optional
            Constant velocity for each boundary
        tthconst : list, optional
            Constant temperature for each boundary
        sthconst : list, optional
            Constant salinity for each boundary
        tobc : list, optional
            Temperature OBC values
        sobc : list, optional
            Salinity OBC values
        tidal_elevations : str or Path, optional
            Path to tidal elevations file
        tidal_velocities : str or Path, optional
            Path to tidal velocities file
        eta_mean, vn_mean : list, optional
            For Flather boundaries (velocity type -1), per open boundary: the mean
            elevation at each node, and the mean normal velocity at each node and
            level. Zero when not given.
        nvrt : int, optional
            Number of vertical levels, for the Flather mean normal velocity, by
            default 2 (a 2D model)
        """
        # Set default values for any None parameters
        flags = flags or [[5, 5, 4, 4]]
        ethconst = ethconst or []
        vthconst = vthconst or []
        tthconst = tthconst or []
        sthconst = sthconst or []
        tobc = tobc or [1]
        sobc = sobc or [1]
        relax = relax or []  # Keep for backward compatibility
        inflow_relax = inflow_relax or [0.5]
        outflow_relax = outflow_relax or [0.1]

        # Assign to instance variables
        self.flags = flags

        # Store tidal file paths
        self.tidal_database = tidal_database
        self.tidal_model = tidal_model
        self.tidal_potential = tidal_potential
        self.cutoff_depth = cutoff_depth
        self.nodal_corrections = nodal_corrections
        self.tide_interpolation_method = normalize_tide_interpolation_method(
            tide_interpolation_method, stacklevel=3
        )
        self.tide_inpaint_iterations = require_inpaint_iterations(
            tide_inpaint_iterations
        )
        self.extrapolate_tides = extrapolate_tides
        self.extrapolation_distance = extrapolation_distance
        self.extra_databases = extra_databases
        self.mdt = mdt
        self._h_coeffs = {}  # Placeholder for harmonic coefficients
        self._uv_coeffs = {}  # Placeholder for UV coefficients

        self.ethconst = ethconst
        self.vthconst = vthconst
        self.tthconst = tthconst
        self.sthconst = sthconst
        self.tobc = tobc
        self.sobc = sobc
        self.relax = relax
        self.inflow_relax = inflow_relax
        self.outflow_relax = outflow_relax
        # Flather boundaries: mean elevation per node and mean normal velocity per
        # node and level, for each open boundary; nvrt is the number of levels
        self.eta_mean = eta_mean
        self.vn_mean = vn_mean
        self.nvrt = nvrt

        # Store boundary condition file paths
        self.elev_th_path = elev_th_path  # Time history of elevation
        self.elev_st_path = elev_st_path  # Space-time elevation
        self.flow_th_path = flow_th_path  # Time history of flow
        self.vel_st_path = vel_st_path  # Space-time velocity
        self.temp_th_path = temp_th_path  # Temperature time history
        self.temp_3d_path = temp_3d_path  # 3D temperature
        self.salt_th_path = salt_th_path  # Salinity time history
        self.salt_3d_path = salt_3d_path  # 3D salinity

        # Store start time and run duration (will be set by SCHISMDataTides.get())
        self._start_time = None
        self._rnday = None

        # Load grid from file or object
        # Assume it's already a grid object
        self.gd = hgrid

        # Define constituent sets (using lowercase for pyTMD compatibility)
        self.major_constituents = ["o1", "k1", "q1", "p1", "m2", "s2", "k2", "n2"]
        self.minor_constituents = ["mm", "mf", "m4", "mn4", "ms4", "2n2", "s1"]

        # Determine which constituents to use
        if isinstance(constituents, str):
            if constituents.lower() == "major":
                self.tnames = self.major_constituents
            elif constituents.lower() == "all":
                self.tnames = self.major_constituents + self.minor_constituents
            else:
                # Assume it's a comma-separated string
                self.tnames = [c.strip() for c in constituents.split(",")]
        elif isinstance(constituents, list):
            self.tnames = constituents
        else:
            # Default to major constituents
            self.tnames = self.major_constituents
        # Unique and lowercase (for pyTMD), in the order given
        self.tnames = list(dict.fromkeys(t.lower() for t in self.tnames))

        # For storing tidal factors
        self.amp = []
        self.freq = []
        self.nodal = []
        self.tear = []
        self.species = []

    @property
    def start_date(self):
        """Get start date for tidal calculations."""
        return self._start_time or datetime.now()

    def _get_tidal_factors(self):
        """Get tidal amplitude, frequency, and species for constituents using pyTMD."""
        if hasattr(self, "amp") and len(self.amp) > 0:
            return
        if not self.tnames:
            self.amp, self.freq, self.species = [], [], []
            self.nodal_factor, self.nodal_phase_correction = [], []
            self.earth_equil_arg = np.array([])
            return
        logger.info(
            f"{ARROW} Computing tidal factors for {len(self.tnames)} constituents"
        )
        # Use pyTMD for all calculations
        ts = timescale.time.Timescale().from_datetime(np.datetime64(self._start_time))
        MJD = ts.MJD
        # Astronomical longitudes
        if self.tidal_model.startswith("FES"):
            # FES models use ASTRO5 method
            s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD, method="ASTRO5")
            u, f = _tmd_args.nodal_modulation(n, p, self.tnames, corrections="FES")
            freq = _tmd_args.frequency(self.tnames, corrections="FES")
        else:
            # Other models use ASTRO2 method
            s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD, method="Cartwright")
            u, f = _tmd_args.nodal_modulation(n, p, self.tnames, corrections="OTIS")
            freq = _tmd_args.frequency(self.tnames, corrections="OTIS")

        # Nodal corrections (u: phase, f: factor), one value per constituent.
        # pyTMD 3 returns a 0-d array for a single constituent; reshape keeps
        # indexing by constituent valid (same fix as upstream main).
        u = np.asarray(u).reshape(-1)
        f = np.asarray(f).reshape(-1)
        u_deg = np.rad2deg(u)

        # Earth equilibrium argument
        hour = 24.0 * np.mod(MJD, 1)
        tau = 15.0 * hour - s + h
        k = 90.0 + np.zeros_like(MJD)
        fargs = np.c_[tau, s, h, p, n, pp, k]
        coef = _tmd_args.coefficients_table(self.tnames)
        G = np.mod(np.dot(fargs, coef), 360.0)

        # Compose info
        self.amp = []
        self.freq = []
        self.nodal_factor = []
        self.nodal_phase_correction = []
        self.species = []
        for c, constituent in enumerate(self.tnames):
            params = _tmd_args._constituent_parameters(constituent)
            self.amp.append(params[0])
            self.freq.append(freq[c])
            self.nodal_factor.append(f[c])
            self.nodal_phase_correction.append(u_deg[c])
            self.species.append(params[4])
        # Store earth equilibrium argument for each constituent
        self.earth_equil_arg = G[0, :]

    def _interpolate_tidal_data(self, lons, lats, constituents, data_type="h"):
        """
        Interpolate tidal harmonics to boundary points via pyTMD 3.

        Opens with ``chunks='auto'`` when dask is installed, then
        ``Dataset.tmd.crop`` on a caller-expanded bounds box (lazy hyperslab;
        no open-time ``bounds`` kwarg).

        Parameters
        ----------
        lons : array
            Longitude values of boundary points
        lats : array
            Latitude values of boundary points
        constituents : list
            Tidal constituent names
        data_type : str
            'h' for elevation, 'uv' for velocity

        Returns
        -------
        np.ndarray
            For elevation: [amp, pha] in meters
            (shape: n_points, n_constituents, 2)
            For velocity: [u_amp, u_pha, v_amp, v_pha] with amplitudes in m/s
            (shape: n_points, n_constituents, 4)
        """
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        tmd_model = pyTMD.io.model(
            self.tidal_database,
            extra_databases=self.extra_databases,
        )
        # Elevation-only databases have no u/v files; pathfinder would FileNotFound.
        groups = ("z", "u", "v") if data_type == "uv" else ("z",)
        model = tmd_model.from_database(self.tidal_model, group=groups)
        if data_type == "h":
            amp, pha = self._interp_group(model, "z", lons, lats, constituents)
            result = np.stack([amp, pha], axis=-1)
        elif data_type == "uv":
            amp_u, pha_u = self._interp_group(model, "u", lons, lats, constituents)
            amp_v, pha_v = self._interp_group(model, "v", lons, lats, constituents)
            result = np.stack([amp_u, pha_u, amp_v, pha_v], axis=-1)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        self._require_finite_harmonics(result, data_type)
        self._warn_if_harmonics_implausible(result, data_type)
        return result

    def _bounds_for_points(self, lons, lats):
        """Envelope of boundary nodes for ``Dataset.tmd.crop``, in degrees.

        pyTMD main after `#613 <https://github.com/pyTMD/pyTMD/pull/613>`_
        shifts each longitude into the model range, so a western-hemisphere
        box on a 0–360 grid (FES, TPXO) still selects cells. A window with
        ``xmin > xmax`` is the short eastward arc across the dateline.
        ``min``/``max`` of longitudes on both sides of ±180 is the long way
        around and reads almost the whole globe.
        """
        # ~111 km per degree of latitude. Longitude degrees shrink by cos(lat),
        # so the same kilometre pad is wider in degrees away from the equator.
        # At least half a degree, so a coarse grid still has an interp halo.
        pad = max(0.5, float(self.extrapolation_distance) / 111.0)
        lons = (np.asarray(lons, dtype=float) + 180.0) % 360.0 - 180.0
        lats = np.asarray(lats, dtype=float)
        lat_rad = np.deg2rad(np.max(np.abs(lats))) if lats.size else 0.0
        lon_pad = pad / max(float(np.cos(lat_rad)), 0.2)
        lon_min = float(np.min(lons))
        lon_max = float(np.max(lons))
        lat_min = float(np.min(lats) - pad)
        lat_max = float(np.max(lats) + pad)
        if lon_max - lon_min > 180.0:
            # Eastward from the western Pacific (positive) across 180.
            return [
                float(np.min(lons[lons >= 0.0]) - lon_pad),
                float(np.max(lons[lons < 0.0]) + lon_pad),
                lat_min,
                lat_max,
            ]
        return [lon_min - lon_pad, lon_max + lon_pad, lat_min, lat_max]

    def _crop_bounds(self, ds, lons, lats):
        """Bounds for ``Dataset.tmd.crop`` in the opened dataset's coordinates.

        Geographic models take the degree envelope from
        :meth:`_bounds_for_points`. Projected models (CATS, AOTIM) take the
        same points transformed into the dataset CRS, padded in metres.
        """
        if bool(ds.tmd.crs.is_geographic):
            return self._bounds_for_points(lons, lats)
        transformed_x, transformed_y = ds.tmd.coords_as(
            np.asarray(lons, dtype=float),
            np.asarray(lats, dtype=float),
            type="drift",
            crs=4326,
        )
        pad_m = max(float(self.extrapolation_distance) * 1000.0, 1000.0)
        x = np.asarray(transformed_x, dtype=float)
        y = np.asarray(transformed_y, dtype=float)
        return [
            float(np.min(x) - pad_m),
            float(np.max(x) + pad_m),
            float(np.min(y) - pad_m),
            float(np.max(y) + pad_m),
        ]

    @staticmethod
    def _resolve_constituent_key(ds, name):
        """Match requested constituent name to a dataset variable."""
        if name in ds:
            return name
        lower = {k.lower(): k for k in ds.data_vars}
        key = lower.get(str(name).lower())
        if key is None:
            raise KeyError(
                f"Constituent {name!r} not in model dataset ({list(ds.data_vars)[:8]}…)"
            )
        return key

    def _xarray_interp_method(self):
        """Map the configured method onto pyTMD 3 / xarray.

        Returns
        -------
        method : str
            ``linear`` or ``nearest`` for ``Dataset.tmd.interp``.
        coastal_fill : bool
            When true, masked model cells are nearest-filled before interp.
        """
        method = normalize_tide_interpolation_method(
            self.tide_interpolation_method or "bilinear",
            stacklevel=3,
        )
        # pyTMD 3 grid_interp forwards `method` to xarray Dataset.interp.
        # On a 2-D grid that is only "linear" and "nearest". "bilinear"
        # is the legacy name for linear after a masked-cell inpaint:
        # pyTMD 2 bilinear kept any finite corner instead of failing the cell.
        if method == "bilinear":
            return "linear", True
        if method in {"linear", "nearest"}:
            return method, False
        logger.warning(
            "Unknown tide_interpolation_method %r; supported values are "
            "%s. Using linear without coastal fill.",
            method,
            ", ".join(sorted(_TIDE_INTERP_METHODS)),
        )
        return "linear", False

    @staticmethod
    def _compute_if_lazy(ds):
        """Load a chunked dataset so in-memory inpaint can run."""
        chunks = getattr(ds, "chunks", None)
        try:
            lazy = chunks is not None and len(chunks) > 0
        except TypeError:
            lazy = False
        if lazy:
            return ds.compute()
        return ds

    def _fill_masked_model_cells(self, ds):
        """Nearest-fill masked tide-model cells inside the cropped window.

        ``Dataset.tmd.inpaint`` (``pyTMD.interpolate.inpaint``) copies the
        nearest finite model node into masked cells and leaves ocean cells
        unchanged. ``tide_inpaint_iterations`` is that function's ``N``:
        0 is nearest neighbors only; ``N > 0`` continues with DCT
        penalized least squares. Interior ``linear`` results therefore
        match a strict linear interpolate when ``N`` is 0. Every masked
        cell in the crop is filled from the nearest wet cell.
        ``extrapolate_tides`` does not limit that fill; the crop pad does.
        Strict ``linear`` / ``nearest`` skip this step, so a stencil that
        touches land stays NaN.
        """
        ds = self._compute_if_lazy(ds)
        geographic = bool(ds.tmd.crs.is_geographic)
        return ds.tmd.inpaint(
            N=self.tide_inpaint_iterations,
            is_geographic=geographic,
        )

    def _require_finite_harmonics(self, result, data_type):
        """Refuse to hand non-finite harmonics to ``bctides.in``."""
        if np.isfinite(result).all():
            return
        node_bad = ~np.isfinite(result).reshape(result.shape[0], -1).all(axis=1)
        n_bad = int(node_bad.sum())
        method = self.tide_interpolation_method or "bilinear"
        if method in ("linear", "nearest"):
            hint = (
                "Nodes may touch a masked coastal cell "
                "(use tide_interpolation_method='bilinear') or lie outside "
                "the tide model (set extrapolate_tides=True or increase "
                "extrapolation_distance)."
            )
        else:
            hint = (
                "Remaining nodes are outside the cropped tide model. "
                "Set extrapolate_tides=True or increase extrapolation_distance. "
                "bilinear fills every masked cell inside that crop, so this "
                "flag does not limit the coastal fill."
            )
        raise ValueError(
            f"{n_bad} boundary node(s) have non-finite {data_type} tidal "
            f"coefficients after interpolation (method={method!r}, "
            f"extrapolate_tides={self.extrapolate_tides}). Refusing to write "
            f"NaN into bctides.in. {hint}"
        )

    def _warn_if_harmonics_implausible(self, result, data_type):
        """Warn when amplitudes cannot be a real tide.

        The OCEANUM test files label elevation as millimetres and transport
        as cm²/s while the stored numbers are already in metres and m²/s.
        pyTMD follows the attributes, so those files come out 10³–10⁴ too
        small. A warning here is the check that the unit test cannot be,
        until those attributes are fixed.
        """
        finite = result[np.isfinite(result)]
        if finite.size == 0:
            return
        if data_type == "h":
            amplitude = np.abs(result[..., 0])
            peak = float(np.nanmax(amplitude))
            if peak == 0.0 or 1.0e-3 <= peak <= 20.0:
                return
            logger.warning(
                "Tidal elevation amplitude peaks at %.3g m, outside 1 mm–20 m. "
                "Check the tide file's units attributes; pyTMD converts from "
                "those, not from a fixed scale.",
                peak,
            )
            return
        amplitude = np.abs(result[..., [0, 2]])
        peak = float(np.nanmax(amplitude))
        if peak <= 10.0:
            return
        logger.warning(
            "Tidal current amplitude peaks at %.3g m/s, above 10 m/s. "
            "Check the tide file's units attributes.",
            peak,
        )

    @staticmethod
    def _currents_to_transport(ds):
        """Undo pyTMD 3's on-grid transport/depth division for OTIS/ATLAS.

        pyTMD 3 opens OTIS/ATLAS ``u``/``v`` groups as transport divided by
        the u/v-node ``bathymetry`` on the model grid. pyTMD 2, OTPS extract_HC
        and TMD 2.5 interpolate transport and depth separately and divide at
        the output point. Multiplying back by ``bathymetry`` recovers the
        transport so :meth:`_transport_to_currents` can divide
        after interpolation. Models without a ``bathymetry`` variable
        (e.g. FES, which stores currents) are returned unchanged.

        This works around pyTMD 3 behaviour and can be removed if pyTMD
        changes. See https://github.com/pyTMD/pyTMD/issues/632.
        pyTMD#624 and pyTMD#625 mask ATLAS nodata to NaN, so land depth
        on ATLAS-netcdf is missing and this divide stays well defined.
        OTIS-binary and TMD3 can still store land depth as 0.
        """
        if "bathymetry" not in ds.data_vars:
            return ds, False
        ds = ds.copy()
        # ATLAS-netcdf on pyTMD main stores land depth as NaN (pyTMD#624,
        # #625). OTIS-binary and TMD3 still store it as 0, which inpaint
        # treats as a real depth and which then inflates coastal currents.
        depth = ds["bathymetry"].where(ds["bathymetry"] > 0)
        ds["bathymetry"] = depth
        for c in ds.tmd.constituents:
            attrs = dict(ds[c].attrs)
            ds[c] = ds[c] * depth
            # The divide in _transport_to_currents cancels this. Do not
            # label it metres: TMD3 stamps transport units onto bathymetry.
            attrs["units"] = "transport"
            ds[c].attrs = attrs
        return ds, True

    @staticmethod
    def _fields_for_interpolation(ds, *, transport):
        """Constituents, plus bathymetry when currents are still transport.

        ``Dataset.tmd.inpaint`` visits every data variable. A fully valid
        integer ``mask`` (TMD3) is not a harmonic, and inpainting it can
        fail or rewrite the mask. Drop everything else before the fill.
        """
        names = list(ds.tmd.constituents)
        if transport and "bathymetry" in ds.data_vars:
            names.append("bathymetry")
        if not names:
            raise ValueError("Tide dataset has no constituent variables")
        return ds[names]

    @staticmethod
    def _transport_to_currents(local):
        """Divide interpolated transport by interpolated depth (m/s)."""
        local = local.copy()
        depth = local["bathymetry"]
        for c in local.tmd.constituents:
            attrs = dict(local[c].attrs)
            local[c] = local[c] / depth
            attrs["units"] = "m/s"
            local[c].attrs = attrs
        return local

    def _interp_group(self, model, group, lons, lats, constituents):
        """Open one tide-model group and interpolate amplitude and phase.

        ``open_dataset`` defaults currents to cm/s. SCHISM ``bctides.in``
        wants m/s, so u/v groups are converted with ``Dataset.tmd.to_units``
        (pyTMD pint). Elevation stays in the default meters.

        ``bilinear`` (the historical default) is xarray ``linear`` after
        :meth:`_fill_masked_model_cells`. Strict ``linear`` / ``nearest``
        do not fill masked cells. ``extrapolate_tides`` still controls
        pyTMD nearest/IDW fill of points that remain outside the model
        grid, limited by ``extrapolation_distance`` kilometres.
        """
        # open_dataset's reduce_constituents defaults to group "z"; reduce
        # the requested group explicitly (needed for u/v).
        model.reduce_constituents(list(constituents), group=group)
        # pyTMD FES: chunked open + tmd.crop (no open-time bounds kwarg).
        ds = model.open_dataset(
            group=group,
            constituents=list(constituents),
            chunks=_OPEN_CHUNKS,
        )
        bounds = self._crop_bounds(ds, lons, lats)
        try:
            ds = ds.tmd.crop(bounds, buffer=0)
        except Exception as exc:
            raise ValueError(
                f"Crop of {self.tidal_model} group {group!r} failed for "
                f"bounds {bounds}: {exc}"
            ) from exc
        transport = False
        if group in ("u", "v"):
            ds = ds.tmd.to_units("m/s")
            ds, transport = self._currents_to_transport(ds)
        ds = self._fields_for_interpolation(ds, transport=transport)

        method, coastal_fill = self._xarray_interp_method()
        if coastal_fill:
            try:
                ds = self._fill_masked_model_cells(ds)
            except Exception as exc:
                raise ValueError(
                    f"Inpaint of {self.tidal_model} group {group!r} failed "
                    f"for bounds {bounds}: {exc}"
                ) from exc
        X, Y = ds.tmd.coords_as(lons, lats, type="drift", crs=4326)
        local = ds.tmd.interp(
            X,
            Y,
            method=method,
            extrapolate=self.extrapolate_tides,
            cutoff=self.extrapolation_distance,
        )
        npts = lons.size
        ncons = len(constituents)
        amp = np.empty((npts, ncons), dtype=float)
        pha = np.empty((npts, ncons), dtype=float)
        if transport:
            local = self._transport_to_currents(local)
        for i, name in enumerate(constituents):
            key = self._resolve_constituent_key(local, name)
            amp[:, i] = np.asarray(local[key].tmd.amplitude).reshape(-1)
            pha[:, i] = np.asarray(local[key].tmd.phase).reshape(-1)
        return amp, pha

    def write_bctides(self, output_file):
        """Generate bctides.in file directly using PyLibs approach.

        Parameters
        ----------
        output_file : str or Path
            Path to output file

        Returns
        -------
        Path
            Path to the created bctides.in file
        """
        # Ensure we have start_time and rnday
        if not self._start_time or self._rnday is None:
            raise ValueError(
                "start_time and rnday must be set before calling write_bctides"
            )

        # Ensure boundary information is computed before accessing boundary attributes
        if hasattr(self.gd, "compute_bnd") and not hasattr(self.gd, "nob"):
            logger.info("Computing boundary information for grid")
            self.gd.compute_bnd()
        elif not hasattr(self.gd, "nob"):
            logger.warning("Grid has no boundary information and no compute_bnd method")

        # Get tidal factors
        self._get_tidal_factors()
        # One crop per group for every open boundary, before the file exists,
        # so a failed interpolate does not leave a partial bctides.in.
        h_by_boundary = self._harmonics_for_open_boundaries("h")
        uv_by_boundary = self._harmonics_for_open_boundaries("uv")

        if self.nodal_corrections:
            logger.info(
                "Applying nodal phase corrections to earth equilibrium argument"
            )
            self.earth_equil_arg = np.mod(
                self.earth_equil_arg + self.nodal_phase_correction, 360.0
            )
        else:
            self.nodal_factor = [1.0] * len(self.tnames)
        output_path = Path(output_file)
        temporary = output_path.with_name(output_path.name + ".tmp")
        try:
            self._write_bctides_file(temporary, h_by_boundary, uv_by_boundary)
            os.replace(temporary, output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return output_path

    def _harmonics_for_open_boundaries(self, data_type):
        """Interpolate one group for every open boundary that uses it.

        Nodes are concatenated, so pyTMD opens and crops the model once
        per group rather than once per boundary.
        """
        flag_index = 0 if data_type == "h" else 1
        wanted = (3, 5) if data_type == "h" else (3, 5, -5)
        pieces = []
        spans = []
        cursor = 0
        for ibnd, bnd_flags in enumerate(self.flags):
            if bnd_flags[flag_index] not in wanted:
                continue
            bnd_nodes = np.asarray(self.gd.iobn[ibnd])
            pieces.append(bnd_nodes)
            spans.append((ibnd, cursor, cursor + bnd_nodes.size))
            cursor += int(bnd_nodes.size)
        if cursor == 0:
            return {}
        all_nodes = np.concatenate(pieces)
        data = self._interpolate_tidal_data(
            self.gd.x[all_nodes],
            self.gd.y[all_nodes],
            self.tnames,
            data_type,
        )
        return {ibnd: data[start:stop] for ibnd, start, stop in spans}

    def _write_bctides_file(self, output_file, h_by_boundary, uv_by_boundary):
        """Write ``bctides.in``. Harmonics are already interpolated."""
        with open(output_file, "w") as f:
            # Write header with date information
            if isinstance(self._start_time, datetime):
                f.write(
                    f"!{self._start_time.month:02d}/{self._start_time.day:02d}/{self._start_time.year:4d} "
                    f"{self._start_time.hour:02d}:00:00 UTC\n"
                )
            else:
                # Assume it's a list [year, month, day, hour]
                year, month, day, hour = self._start_time
                f.write(f"!{month:02d}/{day:02d}/{year:4d} {hour:02d}:00:00 UTC\n")

            # Write tidal potential information
            # Use only constituents with species 0, 1, or 2 (long period, diurnal, semi-diurnal)
            tidal_potential_indices = [
                i for i, s in enumerate(self.species) if s in (0, 1, 2)
            ]
            n_tidal_potential = len(tidal_potential_indices)
            if self.tidal_potential and n_tidal_potential > 0:
                f.write(
                    f" {n_tidal_potential} {self.cutoff_depth:.3f} !number of earth tidal potential, "
                    f"cut-off depth for applying tidal potential\n"
                )
                # Write each constituent's potential information
                for i in tidal_potential_indices:
                    tname = self.tnames[i]
                    species_type = self.species[i]
                    f.write(f"{tname}\n")
                    f.write(
                        f"{species_type} {self.amp[i]:<.6f} {self.freq[i]:<.6e} "
                        f"{self.nodal_factor[i]:.6f} {self.earth_equil_arg[i]:.6f}\n"
                    )
            else:
                # No earth tidal potential
                f.write(
                    " 0 50.000 !number of earth tidal potential, cut-off depth for applying tidal potential\n"
                )

            n_constituents = len(self.tnames)
            if self.mdt is not None:
                # If mdt is provided, we have a constant elevation for all constituents
                n_constituents += 1
            f.write(f"{n_constituents} !nbfr\n")
            if self.mdt is not None:
                # Write mdt as a special constant elevation
                f.write("z0\n")
                f.write("0.0 1.0 0.0\n")

            # Write frequency info for each constituent
            for i, tname in enumerate(self.tnames):
                f.write(
                    f"{tname}\n  {self.freq[i]:<.9e} {self.nodal_factor[i]:7.5f} {self.earth_equil_arg[i]:.5f}\n"
                )

            # Open boundaries: one entry per open boundary of the mesh, in mesh order,
            # with only the values SCHISM reads for each flag (schism_init.F90).
            # Types that read their data from a file (elev.th, elev2D.th.nc, flux.th,
            # uv3D.th.nc, *_3D.th.nc) have nothing else in bctides.in.
            nope = len(self.flags)
            if nope != self.gd.nob:
                raise ValueError(
                    f"bctides.in needs one entry per open boundary: the mesh has "
                    f"{self.gd.nob}, the boundary flags have {nope}"
                )
            f.write(f"{nope} !nope\n")

            for ibnd, bnd_flags in enumerate(self.flags):
                nodes = self.gd.iobn[ibnd]
                num_nodes = self.gd.nobn[ibnd]
                elev_type, vel_type, temp_type, salt_type = bnd_flags
                flag_str = " ".join(map(str, bnd_flags))
                f.write(f"{num_nodes} {flag_str} !open boundary {ibnd + 1}\n")
                lons = self.gd.x[nodes]
                lats = self.gd.y[nodes]

                # Elevation
                if elev_type == 2:
                    f.write(f"{_value(self.ethconst, ibnd, 0.0)} !elevation\n")
                elif elev_type in (3, 5):
                    # If mdt is provided, write the Z0
                    if self.mdt is not None:
                        f.write("z0\n")
                        if isinstance(self.mdt, float):
                            # If mdt is a single float, write it for all nodes
                            for n in range(num_nodes):
                                f.write(f"{self.mdt:.6f} 0.0\n")
                        elif isinstance(self.mdt, (xr.Dataset, xr.DataArray)):
                            # Use a KDTree to efficiently find the closest mdt point for each boundary node
                            mdt_lons = self.mdt.x.values
                            mdt_lats = self.mdt.y.values
                            mdt_values = self.mdt.values
                            # Filter any NaN values in mdt
                            valid_mask = ~np.isnan(mdt_values)
                            mdt_lons = mdt_lons[valid_mask]
                            mdt_lats = mdt_lats[valid_mask]
                            mdt_values = mdt_values[valid_mask]
                            # Create KDTree for mdt points
                            mdt_points = np.column_stack((mdt_lons, mdt_lats))
                            bnd_points = np.column_stack((lons, lats))
                            tree = KDTree(mdt_points)
                            distances, indices = tree.query(bnd_points)
                            tolerance = 0.1
                            if np.any(distances > tolerance):
                                n_pts = np.sum(distances > tolerance)
                                logger.warning(
                                    f"Found {n_pts} boundary points with mdt distance > {tolerance} degrees"
                                )
                            # Extract the mdt values for these points
                            mdt_values = mdt_values[indices]
                            for n in range(num_nodes):
                                mdt_val = float(mdt_values[n])
                                f.write(f"{mdt_val:.6f} 0.0\n")
                        else:
                            # If mdt is not a float or xr.Dataset, raise an error
                            logger.error(
                                f"Invalid mdt type: {type(self.mdt)}. Expected float or xr.Dataset."
                            )

                    logger.info(f"Processing tide for boundary {ibnd + 1}")
                    logger.info(f"Number of boundary nodes: {len(lons)}")
                    logger.info(f"Number of tidal coefficients: {len(self.tnames)}")
                    all_tidal_data = h_by_boundary[ibnd]
                    logger.info(f"Tidal_data shape: {all_tidal_data.shape}")
                    for i, tname in enumerate(self.tnames):
                        # Interpolate tidal data for this constituent
                        try:
                            tidal_data = all_tidal_data[:, i, :].squeeze()
                            if self.nodal_corrections:
                                # Apply nodal correction to phase - amplitude is applied within the code?
                                tidal_data[:, 1] = (
                                    tidal_data[:, 1] + self.nodal_phase_correction[i]
                                ) % 360.0
                            else:
                                # If no nodal corrections, just use the phase as is
                                tidal_data[:, 1] = tidal_data[:, 1] % 360.0

                            # Write header for constituent
                            f.write(f"{tname}\n")

                            # Write amplitude and phase for each node
                            for n in range(num_nodes):
                                f.write(
                                    f"{tidal_data[n, 0]:8.6f} {tidal_data[n, 1]:.6f}\n"
                                )
                        except Exception as e:
                            # Log error but continue with other constituents
                            logger.error(
                                f"Error processing tide {tname} for boundary {ibnd + 1}: {e}"
                            )
                            raise

                # Velocity; relaxation factors come before any tidal constituents
                if vel_type == 2:
                    f.write(f"{_value(self.vthconst, ibnd, 0.0)} !discharge\n")
                if vel_type in (-4, -5):
                    inflow = _value(self.inflow_relax, ibnd, 0.5)
                    outflow = _value(self.outflow_relax, ibnd, 0.1)
                    f.write(f"{inflow} {outflow} !relaxation for inflow, outflow\n")
                if vel_type in (3, 5, -5):
                    if self.mdt is not None:
                        f.write("z0\n")
                        for n in range(num_nodes):
                            f.write("0.0 0.0 0.0 0.0\n")
                    all_vel_data = uv_by_boundary[ibnd]

                    for i, tname in enumerate(self.tnames):
                        # Write header for constituent first
                        f.write(f"{tname}\n")

                        vel_data = all_vel_data[:, i, :].squeeze()

                        if self.nodal_corrections:
                            # Apply nodal correction to phase for u and v components
                            vel_data[:, 1] = (
                                vel_data[:, 1] + self.nodal_phase_correction[i]
                            ) % 360.0
                            vel_data[:, 3] = (
                                vel_data[:, 3] + self.nodal_phase_correction[i]
                            ) % 360.0
                        else:
                            # If no nodal corrections, just use the phases as is
                            vel_data[:, 1] = vel_data[:, 1] % 360.0
                            vel_data[:, 3] = vel_data[:, 3] % 360.0

                        # Write u/v amplitude and phase for each node
                        for n in range(num_nodes):
                            f.write(
                                f"{vel_data[n, 0]:8.6f} {vel_data[n, 1]:.6f} "
                                f"{vel_data[n, 2]:8.6f} {vel_data[n, 3]:.6f}\n"
                            )
                if vel_type == -1:
                    # Flather: mean elevation per node, mean normal velocity per level
                    eta_mean = _value(self.eta_mean, ibnd, None) or [0.0] * num_nodes
                    vn_mean = _value(self.vn_mean, ibnd, None) or [
                        [0.0] * self.nvrt for _ in range(num_nodes)
                    ]
                    f.write("eta_mean\n")
                    for value in eta_mean:
                        f.write(f"{value}\n")
                    f.write("vn_mean\n")
                    for values in vn_mean:
                        f.write(" ".join(str(v) for v in values) + "\n")

                # Temperature, then salinity: a constant value for type 2, then the
                # nudging factor for types 1-4
                for tracer_type, constants, nudging, default in (
                    (temp_type, self.tthconst, self.tobc, 20.0),
                    (salt_type, self.sthconst, self.sobc, 35.0),
                ):
                    if tracer_type == 2:
                        f.write(f"{_value(constants, ibnd, default)}\n")
                    if tracer_type in (1, 2, 3, 4):
                        f.write(f"{_value(nudging, ibnd, 1.0)} !nudging factor\n")


def _value(values, index, default):
    """Return values[index], or default when values is missing or too short."""
    if values is None or index >= len(values) or values[index] is None:
        return default
    return values[index]
