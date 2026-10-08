"""
Tidal boundary conditions for SCHISM.

A direct implementation based on PyLibs scripts/gen_bctides.py with no fallbacks.
"""

from datetime import datetime

import numpy as np
import pyTMD
import timescale
import xarray as xr
from scipy.spatial import KDTree

from rompy.formatting import ARROW
from rompy.logging import get_logger

logger = get_logger(__name__)


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
        extrapolate_tides=False,
        extrapolation_distance=100.0,
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
            Method for tidal interpolation, by default 'bilinear'
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
        self.tide_interpolation_method = tide_interpolation_method
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
            u, f = pyTMD.arguments.nodal_modulation(
                n, p, self.tnames, corrections="FES"
            )
            freq = pyTMD.arguments.frequency(self.tnames, corrections="FES")
        else:
            # Other models use ASTRO2 method
            s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD, method="Cartwright")
            u, f = pyTMD.arguments.nodal_modulation(
                n, p, self.tnames, corrections="OTIS"
            )
            freq = pyTMD.arguments.frequency(self.tnames, corrections="OTIS")

        # Nodal corrections (u: phase, f: factor), one value per constituent
        u = np.asarray(u).reshape(-1)
        f = np.asarray(f).reshape(-1)
        u_deg = np.rad2deg(u)

        # Earth equilibrium argument
        hour = 24.0 * np.mod(MJD, 1)
        tau = 15.0 * hour - s + h
        k = 90.0 + np.zeros_like(MJD)
        fargs = np.c_[tau, s, h, p, n, pp, k]
        coef = pyTMD.arguments.coefficients_table(self.tnames)
        G = np.mod(np.dot(fargs, coef), 360.0)

        # Compose info
        self.amp = []
        self.freq = []
        self.nodal_factor = []
        self.nodal_phase_correction = []
        self.species = []
        for c, constituent in enumerate(self.tnames):
            params = pyTMD.arguments._constituent_parameters(constituent)
            self.amp.append(params[0])
            self.freq.append(freq[c])
            self.nodal_factor.append(f[c])
            self.nodal_phase_correction.append(u_deg[c])
            self.species.append(params[4])
        # Store earth equilibrium argument for each constituent
        self.earth_equil_arg = G[0, :]

    def _interpolate_tidal_data(self, lons, lats, constituents, data_type="h"):
        """
        Interpolate tidal data for a constituent to boundary points using pyTMD extract_constants.

        Parameters
        ----------
        lons : array
            Longitude values of boundary points
        lats : array
            Latitude values of boundary points
        constituent : str
            Tidal constituent name
        data_type : str
            'h' for elevation, 'uv' for velocity

        Returns
        -------
        np.ndarray
            For elevation: [amp, pha] (shape: n_points, 2)
            For velocity: [u_amp, u_pha, v_amp, v_pha] (shape: n_points, 4)
        """
        tmd_model = pyTMD.io.model(
            self.tidal_database,
            extra_databases=self.extra_databases,
            constituents=constituents,
        )
        if data_type == "h":
            amp, pha, _ = tmd_model.elevation(self.tidal_model).extract_constants(
                lons,
                lats,
                constituents=constituents,
                method=self.tide_interpolation_method,
                crop=True,
                extrapolate=self.extrapolate_tides,
                cutoff=self.extrapolation_distance,
            )
            # (points, constituents, 1), also for a single point or constituent
            amp = np.asarray(amp).reshape(len(lons), -1)[..., None]
            pha = np.asarray(pha).reshape(len(lons), -1)[..., None]
            # Return shape (n_points, 2)
            return np.concatenate((amp, pha), axis=-1)
        elif data_type == "uv":
            amp_u, pha_u, _ = tmd_model.current(self.tidal_model).extract_constants(
                lons,
                lats,
                type="u",
                constituents=constituents,
                method=self.tide_interpolation_method,
                crop=True,
                extrapolate=self.extrapolate_tides,
                cutoff=self.extrapolation_distance,
            )
            amp_v, pha_v, _ = tmd_model.current(self.tidal_model).extract_constants(
                lons,
                lats,
                type="v",
                constituents=constituents,
                method=self.tide_interpolation_method,
                crop=True,
                extrapolate=self.extrapolate_tides,
                cutoff=self.extrapolation_distance,
            )
            # (points, constituents, 1); pyTMD returns currents in cm/s
            shape = (len(lons), -1)
            amp_u = (np.asarray(amp_u).reshape(shape) / 100)[..., None]
            pha_u = np.asarray(pha_u).reshape(shape)[..., None]
            amp_v = (np.asarray(amp_v).reshape(shape) / 100)[..., None]
            pha_v = np.asarray(pha_v).reshape(shape)[..., None]
            # Return shape (n_points, 4)
            return np.concatenate((amp_u, pha_u, amp_v, pha_v), axis=-1)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

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

        if self.nodal_corrections:
            logger.info(
                "Applying nodal phase corrections to earth equilibrium argument"
            )
            self.earth_equil_arg = np.mod(
                self.earth_equil_arg + self.nodal_phase_correction, 360.0
            )
        else:
            self.nodal_factor = [1.0] * len(self.tnames)
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

                    logger.info(f"Processing tide for boundary {ibnd+1}")
                    logger.info(f"Number of boundary nodes: {len(lons)}")
                    logger.info(f"Number of tidal coefficients: {len(self.tnames)}")
                    all_tidal_data = self._interpolate_tidal_data(
                        lons, lats, self.tnames, "h"
                    )
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
                                    f"{tidal_data[n,0]:8.6f} {tidal_data[n,1]:.6f}\n"
                                )
                        except Exception as e:
                            # Log error but continue with other constituents
                            logger.error(
                                f"Error processing tide {tname} for boundary {ibnd+1}: {e}"
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
                    all_vel_data = self._interpolate_tidal_data(
                        lons, lats, self.tnames, "uv"
                    )

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
                                f"{vel_data[n,0]:8.6f} {vel_data[n,1]:.6f} "
                                f"{vel_data[n,2]:8.6f} {vel_data[n,3]:.6f}\n"
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

        return output_file


def _value(values, index, default):
    """Return values[index], or default when values is missing or too short."""
    if values is None or index >= len(values) or values[index] is None:
        return default
    return values[index]
