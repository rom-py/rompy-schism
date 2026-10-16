"""Concrete expected outputs and filesystem evidence for SCHISM."""

import pytest
from rompy.core.responses import ArtifactType

from rompy_schism.config import SCHISMConfig
from rompy_schism.config_legacy import SchismCSIROConfig
from rompy_schism.namelists import NML
from rompy_schism.namelists.param import Param


def config(*, core=None, schout=None, opt=None):
    values = {"dt": 3600, "rnday": 1, "nspool": 2, "ihfskip": 12}
    values.update(core or {})
    param = Param(core=values, schout=schout or {}, opt=opt or {})
    return SCHISMConfig.model_construct(nml=NML(param=param))


def paths(cfg):
    return {a.path for a in cfg.expected_artifacts()}


def test_default_hydro_stacks():
    expected = paths(config())
    assert expected == {"outputs/mirror.out"} | {
        f"outputs/{name}_{stack}.nc"
        for name in ("out2d", "zCoordinates", "horizontalVelX", "horizontalVelY")
        for stack in (1, 2)
    }
    assert not any("*" in path for path in expected)


@pytest.mark.parametrize(
    "steps,stacks",
    [(1, []), (2, [1]), (12, [1]), (13, [1]), (14, [1, 2]), (24, [1, 2])],
)
def test_partial_stack_requires_output_record(steps, stacks):
    expected = paths(config(core={"rnday": steps / 24}))
    assert {p for p in expected if "/out2d_" in p} == {
        f"outputs/out2d_{stack}.nc" for stack in stacks
    }


def test_output_disabled():
    assert paths(config(schout={"nc_out": 0})) == {"outputs/mirror.out"}


def test_selective_3d_outputs():
    expected = paths(config(schout={"iof_hydro__18": 1, "iof_hydro__26": 0}))
    assert "outputs/temperature_1.nc" in expected
    assert "outputs/horizontalVelX_1.nc" not in expected
    assert "outputs/salinity_1.nc" not in expected


def test_hotstart_rank_zero_is_concrete():
    cfg = config(schout={"nhot": 1, "nhot_write": 12, "nc_out": 0})
    assert paths(cfg) == {
        "outputs/mirror.out",
        "outputs/hotstart_000000_12.nc",
        "outputs/hotstart_000000_24.nc",
    }
    assert cfg.expected_artifacts()[1].artifact_type == ArtifactType.RESTART


@pytest.mark.parametrize("kwargs", [{"core": {"ipre": 1}}, {"opt": {"ihot": 2}}])
def test_unknown_or_preprocessing_clock_is_not_guessed(kwargs):
    assert paths(config(**kwargs)) == {"outputs/mirror.out"}


def test_no_namelist():
    assert paths(SCHISMConfig.model_construct(nml=None)) == {"outputs/mirror.out"}


def test_legacy_config_has_explicit_contract():
    assert paths(SchismCSIROConfig.model_construct()) == {"outputs/mirror.out"}


def test_core_generation_execution_and_sidecar(grid2d, tmp_path):
    from rompy.backends import LocalConfig
    from rompy.core.result_persistence import load_run_result
    from rompy.model import ModelRun

    cfg = SCHISMConfig(grid=grid2d)
    run = ModelRun(
        run_id="schema-test",
        config=cfg,
        output_dir=str(tmp_path),
        period={
            "start": "2024-01-01",
            "end": "2024-01-01T01:00:00",
            "interval": "100s",
        },
    )
    workspace = tmp_path / run.run_id
    workspace.mkdir()
    # Real input generation and core result handling; no SCHISM executable.
    with pytest.warns(UserWarning, match="out2d_1.nc"):
        result = run.run(
            backend=LocalConfig(
                command="printf log > outputs/mirror.out",
                working_dir=workspace,
            )
        )
    assert result.success
    assert "outputs/out2d_1.nc" in {a.path for a in result.missing_outputs}
    assert "outputs/mirror.out" in {a.path for a in result.artifacts}
    assert load_run_result(workspace).payload == result


def test_observations_include_mpi_and_stations_not_missing(tmp_path):
    cfg = config(schout={"nc_out": 0, "nhot": 1, "nhot_write": 12})
    output = tmp_path / "outputs"
    output.mkdir()
    for name in (
        "mirror.out",
        "staout_1",
        "hotstart_000000_12.nc",
        "hotstart_000001_12.nc",
    ):
        (output / name).write_text("data")
    (tmp_path / "run_result.json").write_text("{}")
    with pytest.warns(UserWarning, match="hotstart_000000_24.nc"):
        observed = cfg.validate_outputs(tmp_path)
    assert {a.path for a in observed} == {
        f"outputs/{name}"
        for name in (
            "mirror.out",
            "staout_1",
            "hotstart_000000_12.nc",
            "hotstart_000001_12.nc",
        )
    }
    assert all(a.size_bytes == 4 for a in observed)
    assert all(
        a.artifact_type == ArtifactType.RESTART
        for a in observed
        if "hotstart" in a.path
    )
