"""One-shot FES interp: all tidal opens in one pyTMD crop, not per-boundary."""

from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from rompy_schism.bctides import Bctides


class _MockGrid:
    def __init__(self):
        self.ne = 20
        self.np = 20
        self.nob = 3
        self.nobn = np.array([4, 5, 3], dtype=np.int32)
        self.iobn = [
            np.arange(0, 4, dtype=np.int32),
            np.arange(4, 9, dtype=np.int32),
            np.arange(9, 12, dtype=np.int32),
        ]
        self.x = np.linspace(145.0, 146.0, 20)
        self.y = np.linspace(-16.0, -15.0, 20)


def test_write_bctides_interpolates_once_per_group():
    grid = _MockGrid()
    bctides = Bctides(
        hgrid=grid,
        flags=[[3, 3, 0, 0], [3, 3, 0, 0], [3, 3, 0, 0]],
        constituents=["M2", "S2"],
        tidal_database="/unused",
        tidal_model="FES2022",
        nodal_corrections=False,
        tidal_potential=False,
    )
    bctides._start_time = datetime(2023, 1, 1)
    bctides._rnday = 2.0
    bctides.ethconst = {}
    bctides.vthconst = {}

    calls = []

    def mock_interpolate(lons, lats, tnames, data_type):
        calls.append((data_type, int(np.size(lons))))
        n = len(np.atleast_1d(lons))
        ncons = len(tnames)
        if data_type == "h":
            arr = np.zeros((n, ncons, 2))
            arr[..., 0] = 0.5
            arr[..., 1] = 45.0
            return arr
        arr = np.zeros((n, ncons, 4))
        arr[..., 0] = 0.1
        arr[..., 1] = 30.0
        arr[..., 2] = 0.1
        arr[..., 3] = 60.0
        return arr

    bctides._interpolate_tidal_data = mock_interpolate

    with NamedTemporaryFile(delete=False) as tmp:
        path = Path(tmp.name)
    try:
        bctides.write_bctides(path)
    finally:
        path.unlink(missing_ok=True)

    kinds = [c[0] for c in calls]
    assert kinds.count("h") == 1, calls
    assert kinds.count("uv") == 1, calls
    assert calls[0][1] == 12
    assert calls[1][1] == 12
