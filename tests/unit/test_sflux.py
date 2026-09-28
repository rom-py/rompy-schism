"""sflux atmospheric forcing."""

import numpy as np
import pytest
from pydantic import ValidationError
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.core.types import DatasetCoords

from rompy_schism.data import SCHISMDataSflux, SfluxAir, SfluxPrc, SfluxRad

COORDS = DatasetCoords(t="time", x="longitude", y="latitude")
FILTER = {"sort": {"coords": ["latitude"]}}


def era5_air(test_files_dir, **kwargs) -> SfluxAir:
    return SfluxAir(
        source=SourceFile(uri=str(test_files_dir / "era5.nc")),
        coords=COORDS,
        filter=FILTER,
        uwind_name="u10",
        vwind_name="v10",
        **kwargs,
    )


def test_missing_air_variables_are_a_standard_atmosphere(test_files_dir):
    ds = era5_air(test_files_dir).ds
    assert np.all(ds.prmsl == 101325.0)
    assert np.all(ds.stmp == 288.15)
    assert np.all(ds.spfh == 0.01)
    assert ds.prmsl.shape == ds.u10.shape


def test_forcing_period_is_padded_once_for_all_variables(
    test_files_dir, grid2d, tmp_path, monkeypatch
):
    periods = []
    monkeypatch.setattr(
        SfluxAir, "get", lambda self, destdir, grid, time: periods.append(time)
    )
    sflux = SCHISMDataSflux(
        air_1=era5_air(test_files_dir, relative_weight=0.5),
        air_2=era5_air(test_files_dir, relative_weight=0.5),
    )
    period = TimeRange(start="2023-01-02T00", end="2023-01-03T00", interval="1h")
    sflux.get(tmp_path, grid2d, period)
    padded = TimeRange(start="2023-01-01T00", end="2023-01-04T00", interval="1h")
    assert periods == [padded, padded]


def test_relative_weights_are_checked_for_every_kind():
    # Only sources with fail_if_missing=False are checked
    rad = SfluxRad(
        source=SourceFile(uri="rad.nc"),
        dlwrf_name="strd",
        relative_weight=0.5,
        fail_if_missing=False,
    )
    with pytest.raises(ValidationError, match="Relative weights for rad"):
        SCHISMDataSflux(rad_1=rad)


def test_precipitation_data_type():
    assert SfluxPrc(source=SourceFile(uri="prc.nc"), prate_name="tp").data_type == (
        "sflux_prc"
    )
