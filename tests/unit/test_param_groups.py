"""param.nml is written with the three namelist groups SCHISM reads."""

import re

import pytest
from pydantic import ValidationError

from rompy_schism.namelists.param import Opt, Param


def groups(rendered: str) -> dict[str, str]:
    """Split a rendered namelist into {group name: body}."""
    return {
        name.lower(): body
        for name, body in re.findall(r"^&(\w+)\n(.*?)^/", rendered, re.DOTALL | re.MULTILINE)
    }


def test_param_has_only_the_groups_schism_reads():
    assert list(groups(Param().render())) == ["core", "opt", "schout"]


@pytest.mark.parametrize("name", ["s1_mxnbt", "rho0", "iflux", "iveg", "veg_cw"])
def test_former_vertical_and_vegetation_parameters_are_in_opt(name):
    opt = groups(Param(opt={name: Opt().model_dump()[name]}).render())["opt"]
    assert re.search(rf"^{name} = ", opt, re.MULTILINE)


def test_deprecated_sections_move_to_opt():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        param = Param(
            opt=Opt(ihot=1),
            vertical={"s1_mxnbt": 0.77},
            VEGETATION={"VEG_CW": 2},
        )
    assert (param.opt.ihot, param.opt.s1_mxnbt, param.opt.veg_cw) == (1, 0.77, 2)
    assert "s1_mxnbt = 0.77" in groups(param.render())["opt"]


def test_deprecated_section_conflicting_with_opt_raises():
    with pytest.warns(DeprecationWarning), pytest.raises(ValidationError):
        Param(opt={"s1_mxnbt": 0.5}, vertical={"s1_mxnbt": 0.6})


def test_vegetation_coefficients_are_integers():
    # SCHISM v5.13/v5.14 declare them as integers; "1.5" is a fatal read error
    opt = groups(Param().render())["opt"]
    assert re.search(r"^veg_lai = 1$", opt, re.MULTILINE)
    assert re.search(r"^veg_cw = 1$", opt, re.MULTILINE)
    with pytest.raises(ValidationError):
        Opt(veg_cw=1.5)


def test_hotstart_output_interval_is_checked_across_groups():
    Param(core={"ihfskip": 720}, schout={"nhot": 1, "nhot_write": 1440})
    with pytest.raises(ValidationError, match="multiple of core.ihfskip"):
        Param(core={"ihfskip": 720}, schout={"nhot": 1, "nhot_write": 1000})
    with pytest.raises(ValidationError, match="multiple of schout.nspool_sta"):
        Param(schout={"iout_sta": 1, "nspool_sta": 7, "nhot_write": 1000})
