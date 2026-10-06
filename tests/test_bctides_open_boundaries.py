"""bctides.in has one entry per open boundary, written as SCHISM reads it."""

import pytest
import xarray as xr
from rompy.core.data import DataBlob
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords

from rompy_schism.boundary_core import TidalDataset
from rompy_schism.data import (
    BoundarySetupWithSource,
    SCHISMDataBoundary,
    SCHISMDataBoundaryConditions,
)
from rompy_schism.grid import SCHISMGrid

TIDAL = BoundarySetupWithSource(elev_type=3, vel_type=3)
PERIOD = TimeRange(start="2023-01-01T00", end="2023-01-02T00", interval="1h")


@pytest.fixture
def grid_two_open_boundaries(tmp_path):
    """A 3x3-node mesh with open boundaries on its west and east sides."""
    nodes = [(150.0 + 0.1 * i, -20.0 + 0.1 * j) for j in range(3) for i in range(3)]
    elements = []
    for j in range(2):
        for i in range(2):
            a, b = j * 3 + i + 1, j * 3 + i + 2
            c, d = b + 3, a + 3
            elements += [(a, b, c), (a, c, d)]
    lines = ["mesh", f"{len(elements)} {len(nodes)}"]
    lines += [f"{n + 1} {x} {y} 10.0" for n, (x, y) in enumerate(nodes)]
    lines += [f"{e + 1} 3 {a} {b} {c}" for e, (a, b, c) in enumerate(elements)]
    lines += [
        "2 = Number of open boundaries",
        "6 = Total number of open boundary nodes",
        "3 = Number of nodes for open boundary 1",
        "7", "4", "1",
        "3 = Number of nodes for open boundary 2",
        "3", "6", "9",
        "2 = number of land boundaries",
        "6 = Total number of land boundary nodes",
        "3 0 = Number of nodes for land boundary 1",
        "1", "2", "3",
        "3 0 = Number of nodes for land boundary 2",
        "9", "8", "7",
    ]  # fmt: skip
    hgrid = tmp_path / "hgrid.gr3"
    hgrid.write_text("\n".join(lines) + "\n")
    return SCHISMGrid(hgrid=DataBlob(source=hgrid), drag=0.0025)


def bctides(conditions, grid, tmp_path) -> list[str]:
    """Write bctides.in and return its lines without the comments."""
    conditions.get(tmp_path, grid, PERIOD)
    text = (tmp_path / "bctides.in").read_text()
    return [line.split("!")[0].strip() for line in text.splitlines()[1:]]


def boundary_lines(lines, nbfr) -> list[str]:
    """Lines after the constituent header (ntip line, nbfr line, 2 per constituent)."""
    return lines[2 + 2 * nbfr :]


def tides(tidal_data_files, constituents=("M2", "S2")):
    return TidalDataset(
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
        constituents=list(constituents),
        tidal_potential=False,
        mean_dynamic_topography=None,
        extrapolate_tides=True,
        extrapolation_distance=1000.0,
    )


def test_one_entry_per_open_boundary(
    grid_two_open_boundaries, tidal_data_files, tmp_path
):
    conditions = SCHISMDataBoundaryConditions(
        tidal_data=tides(tidal_data_files), default_boundary=TIDAL
    )
    lines = boundary_lines(bctides(conditions, grid_two_open_boundaries, tmp_path), 2)
    assert lines[0] == "2"  # nope
    assert lines[1] == "3 3 3 0 0"  # boundary 1: 3 nodes, tidal elevation and currents
    assert lines.count("3 3 3 0 0") == 2


def test_open_boundary_without_setup_raises(
    grid_two_open_boundaries, tidal_data_files, tmp_path
):
    conditions = SCHISMDataBoundaryConditions(
        tidal_data=tides(tidal_data_files), boundaries={0: TIDAL}
    )
    with pytest.raises(
        ValueError, match=r"Open boundaries \[1\] of the mesh have no setup"
    ):
        conditions.get(tmp_path, grid_two_open_boundaries, PERIOD)


def test_unknown_open_boundary_raises(
    grid_two_open_boundaries, tidal_data_files, tmp_path
):
    conditions = SCHISMDataBoundaryConditions(
        tidal_data=tides(tidal_data_files),
        boundaries={2: TIDAL},
        default_boundary=TIDAL,
    )
    with pytest.raises(ValueError, match=r"boundaries \[2\] are not open boundaries"):
        conditions.get(tmp_path, grid_two_open_boundaries, PERIOD)


def test_file_types_have_only_nudging_factors(grid_two_open_boundaries, tmp_path):
    # Elevation, velocity, temperature and salinity all from *.th.nc files: SCHISM
    # reads only the two tracer nudging factors after each flag line
    setup = BoundarySetupWithSource(elev_type=4, vel_type=4, temp_type=4, salt_type=4)
    conditions = SCHISMDataBoundaryConditions(default_boundary=setup)
    lines = bctides(conditions, grid_two_open_boundaries, tmp_path)
    assert lines[:2] == ["0 50.000", "0"]  # no tidal potential, no constituents
    assert boundary_lines(lines, 0) == [
        "2", "3 4 4 4 4", "1.0", "1.0", "3 4 4 4 4", "1.0", "1.0",
    ]  # fmt: skip


def test_constant_types_have_one_value(grid_two_open_boundaries, tmp_path):
    setup = BoundarySetupWithSource(
        elev_type=2, vel_type=2, temp_type=2, salt_type=2,
        const_elev=0.5, const_flow=-100.0, const_temp=18.0, const_salt=34.0,
    )  # fmt: skip
    conditions = SCHISMDataBoundaryConditions(default_boundary=setup)
    lines = boundary_lines(bctides(conditions, grid_two_open_boundaries, tmp_path), 0)
    assert lines[1:8] == ["3 2 2 2 2", "0.5", "-100.0", "18.0", "1.0", "34.0", "1.0"]


def test_flather_has_a_mean_velocity_per_level(grid_two_open_boundaries, tmp_path):
    setup = BoundarySetupWithSource(elev_type=0, vel_type=-1)
    conditions = SCHISMDataBoundaryConditions(default_boundary=setup)
    lines = boundary_lines(bctides(conditions, grid_two_open_boundaries, tmp_path), 0)
    # 2D model: eta_mean per node, then vn_mean with 2 levels per node
    assert lines[1:10] == [
        "3 0 -1 0 0", "eta_mean", "0.0", "0.0", "0.0",
        "vn_mean", "0.0 0.0", "0.0 0.0", "0.0 0.0",
    ]  # fmt: skip


def test_constituents_keep_their_order(
    grid_two_open_boundaries, tidal_data_files, tmp_path
):
    conditions = SCHISMDataBoundaryConditions(
        tidal_data=tides(tidal_data_files, ["S2", "N2", "M2"]), default_boundary=TIDAL
    )
    lines = bctides(conditions, grid_two_open_boundaries, tmp_path)
    assert [lines[2], lines[4], lines[6]] == ["s2", "n2", "m2"]


def test_single_constituent(grid_two_open_boundaries, tidal_data_files, tmp_path):
    conditions = SCHISMDataBoundaryConditions(
        tidal_data=tides(tidal_data_files, ["M2"]), default_boundary=TIDAL
    )
    lines = bctides(conditions, grid_two_open_boundaries, tmp_path)
    assert lines[1:3] == ["1", "m2"]


def elevation_source(test_files_dir):
    return SCHISMDataBoundary(
        source=SourceFile(uri=str(test_files_dir / "hycom.nc")),
        variables=["surf_el"],
        coords=DatasetCoords(t="time", x="xlon", y="ylat"),
    )


def test_boundary_file_holds_the_boundaries_that_use_it(
    grid_two_open_boundaries, tidal_data_files, test_files_dir, tmp_path
):
    ocean = BoundarySetupWithSource(
        elev_type=4, vel_type=0, elev_source=elevation_source(test_files_dir)
    )
    conditions = SCHISMDataBoundaryConditions(
        tidal_data=tides(tidal_data_files),
        boundaries={1: ocean},
        default_boundary=TIDAL,
    )
    lines = boundary_lines(bctides(conditions, grid_two_open_boundaries, tmp_path), 2)
    assert lines[1] == "3 3 3 0 0"  # boundary 1: tides
    assert "3 4 0 0 0" in lines  # boundary 2: elev2D.th.nc
    elev2d = xr.open_dataset(tmp_path / "elev2D.th.nc")
    assert elev2d.sizes["nOpenBndNodes"] == 3  # only the nodes of boundary 2


def test_equivalent_sources_can_share_boundary_file(
    grid_two_open_boundaries, test_files_dir, tmp_path
):
    conditions = SCHISMDataBoundaryConditions(
        boundaries={
            i: BoundarySetupWithSource(
                elev_type=4, vel_type=0, elev_source=elevation_source(test_files_dir)
            )
            for i in (0, 1)
        }
    )
    conditions.get(tmp_path, grid_two_open_boundaries, PERIOD)
    assert xr.open_dataset(tmp_path / "elev2D.th.nc").sizes["nOpenBndNodes"] == 6


def test_distinct_sources_cannot_share_boundary_file(
    grid_two_open_boundaries, test_files_dir, tmp_path
):
    first = elevation_source(test_files_dir)
    second = SCHISMDataBoundary(
        source=SourceFile(uri=str(test_files_dir / "different.nc")),
        variables=["surf_el"],
        coords=DatasetCoords(t="time", x="xlon", y="ylat"),
    )
    conditions = SCHISMDataBoundaryConditions(
        boundaries={
            0: BoundarySetupWithSource(elev_type=4, vel_type=0, elev_source=first),
            1: BoundarySetupWithSource(elev_type=4, vel_type=0, elev_source=second),
        }
    )
    with pytest.raises(ValueError, match="need the same elev_source"):
        conditions.get(tmp_path, grid_two_open_boundaries, PERIOD)
