"""Tests for versioned SCHISM CORE/OPT ``param.nml`` contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rompy_schism.namelists.param import (
    Core,
    CoreV513,
    CoreV514,
    Opt,
    OptV513,
    OptV514,
    Param,
    ParamV513,
    ParamV514,
)
from rompy_schism.namelists.schism import NML

pytest.importorskip("rompy_schism")


class TestParamV513:
    def test_legacy_procedural_proxies(self):
        assert issubclass(Core, CoreV513)
        assert issubclass(Opt, OptV513)
        assert issubclass(Param, ParamV513)
        assert Core.__name__ == "Core"
        assert Opt.__name__ == "Opt"
        assert Param.__name__ == "Param"

    def test_legacy_rendering(self):
        text = Param().render()
        assert "isconsv = 0" in text
        assert "nmarsh_types" not in text
        assert "param_schema" not in text

    def test_yaml_without_schema_defaults_to_v513(self):
        nml = NML.model_validate({"param": {"core": {"dt": 150}}})
        assert isinstance(nml.param, ParamV513)
        assert nml.param.param_schema == "schism-v5.13"
        assert nml.param.core.dt == 150

    def test_legacy_isconsv_validation(self):
        assert Opt(isconsv=1).isconsv == 1
        with pytest.raises(ValidationError):
            Opt(isconsv=2)


class TestParamV514:
    def test_explicit_schema_deserializes_to_v514(self):
        nml = NML.model_validate({"param": {"param_schema": "schism-v5.14"}})
        assert isinstance(nml.param, ParamV514)

    def test_modern_rendering(self):
        text = ParamV514().render()
        assert "nmarsh_types = 2" in text
        assert "isconsv =" not in text
        assert "param_schema" not in text

    def test_nmarsh_types_default(self):
        assert CoreV514().nmarsh_types == 2

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nmarsh_types_must_be_positive(self, bad):
        with pytest.raises(ValidationError):
            CoreV514(nmarsh_types=bad)

    def test_modern_opt_rejects_isconsv(self):
        assert "isconsv" not in OptV514.model_fields
        with pytest.raises(ValidationError, match="isconsv"):
            OptV514(isconsv=0)

    def test_versioned_class_still_writes_param_filename(self, tmp_path):
        ParamV514().write_nml(tmp_path)
        assert (Path(tmp_path) / "param.nml").is_file()
        assert not (Path(tmp_path) / "paramv514.nml").exists()


class TestParamSchemaConversion:
    def test_convert_legacy_to_modern(self):
        modern = Param(opt=Opt(isconsv=0)).to_schema("schism-v5.14")
        assert isinstance(modern, ParamV514)
        assert modern.core.nmarsh_types == 2

    def test_convert_modern_to_legacy(self):
        legacy = ParamV514(core={"nmarsh_types": 3}).to_schema("schism-v5.13")
        assert isinstance(legacy, ParamV513)
        assert legacy.opt.isconsv == 0

    def test_refuse_lossy_isconsv_conversion(self):
        with pytest.raises(ValueError, match="PREC_EVAP"):
            Param(opt=Opt(isconsv=1)).to_schema("schism-v5.14")

    def test_nested_update_preserves_modern_schema(self):
        nml = NML(param=ParamV514())
        nml.update({"param": {"core": {"dt": 75}}})
        assert isinstance(nml.param, ParamV514)
        assert nml.param.core.dt == 75
