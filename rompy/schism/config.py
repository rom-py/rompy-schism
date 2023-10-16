import logging
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from rompy.core import BaseConfig, DataBlob, RompyBaseModel, Spectrum, TimeRange

from .grid import SCHISMGrid2D

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

CSIRO_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "schism")


class Inputs(RompyBaseModel):
    filewave: DataBlob | None = Field(
        None,
        description="TODO",
    )

    def get(self, staging_dir: Path):
        ret = {}
        for source in self:
            ret[source[0]] = source[1].get(staging_dir).name
        return ret

    def __str__(self):
        ret = ""
        for forcing in self:
            if forcing[1]:
                ret += f"\t{forcing[0]}: {forcing[1].source}\n"
        return ret


class SchismCSIROConfig(BaseConfig):
    model_type: Literal["schismcsiro"] = Field(
        "schismcsiro", description="The model type for SCHISM."
    )
    grid: SCHISMGrid2D = Field(description="The model grid")
    inputs: Inputs = Field(description="Model inputs")
    project: str = Field("WAXA", description="TODO")
    utc_start: int = Field(0, description="TODO")
    rnday: int = Field(120, description="TODO")
    time_step: float = Field(120.0, description="TODO")
    msc2: int = Field(36, description="TODO")
    mdc2: int = Field(36, description="TODO")
    ihfskip: int = Field(720, description="TODO")
    icou_elfe_wwm: int = Field(1, description="TODO")
    nstep_wwm: int = Field(3, description="TODO")
    deltc: int = Field(360, description="TODO")
    h1_bcc: int = Field(50, description="TODO")
    h2_bcc: int = Field(100, description="TODO")
    h_bcc1: int = Field(100, description="TODO")
    thetai: float = Field(0.8, description="TODO")
    iwbl: int = Field(0, description="TODO")
    slam0: int = Field(120, description="TODO")
    sfea0: int = Field(-29, description="TODO")
    nchi: int = Field(-1, description="TODO")
    dzb_decayYN: str = Field("!", description="TODO")
    rlatitude: int = Field(-29, description="TODO")
    ic_elev: int = Field(0, description="TODO")
    inv_atm_bnd: int = Field(1, description="TODO")
    ibtrack_openbndYN: str = Field("!", description="TODO")
    iwindoffYN: str = Field("!", description="TODO")
    iwind_form: int = Field(1, description="TODO")
    sav_cdYN: str = Field("!", description="TODO")
    iout_sta: int = Field(0, description="TODO")
    lindsprdeg: str = Field("F", description="TODO")
    wbdm: int = Field(90, description="TODO")
    extrapYN: str = Field("!", description="TODO")
    extrap: str = Field("T", description="TODO")
    windYN: str = Field("!", description="TODO")
    filewind: str = Field("wind.dat", description="TODO")
    currYN: str = Field("!", description="TODO")
    walvYN: str = Field("!", description="TODO")
    mesin: int = Field(1, description="TODO")
    mesbf: int = Field(2, description="TODO")
    fricc: float = Field(0.11, description="TODO")
    ibreak: int = Field(1, description="TODO")
    brcrYN: str = Field("", description="TODO")
    melim: int = Field(1, description="TODO")
    limfak: float = Field(0.1, description="TODO")
    lsourceswam: str = Field("F", description="TODO")
    deltc_out: int = Field(3600, description="TODO")
    definetc: int = Field(-1, description="TODO")
    outstyle: str = Field("NC", description="TODO")
    wwm1: int = Field(1, description="TODO")
    HS: str = Field("T", description="TODO")
    wwm2: int = Field(1, description="TODO")
    TM01: str = Field("T", description="TODO")
    wwm3: int = Field(0, description="TODO")
    TM02: str = Field("F", description="TODO")
    DM: str = Field("T", description="TODO")
    wwm4: int = Field(0, description="TODO")
    wwm5: int = Field(0, description="TODO")
    wwm6: int = Field(0, description="TODO")
    wwm7: int = Field(0, description="TODO")
    wwm8: int = Field(1, description="TODO")
    DSPR: str = Field("T", description="TODO")
    wwm9: int = Field(1, description="TODO")
    TPP: str = Field("T", description="TODO")
    wwm10: int = Field(0, description="TODO")
    wwm11: int = Field(0, description="TODO")
    wwm12: int = Field(0, description="TODO")
    wwm13: int = Field(0, description="TODO")
    wwm14: int = Field(0, description="TODO")
    wwm15: int = Field(0, description="TODO")
    wwm16: int = Field(1, description="TODO")
    PEAKD: str = Field("T", description="TODO")
    DPEAK: str = Field("T", description="TODO")
    wwm17: int = Field(1, description="TODO")
    PEAKDSPR: str = Field("T", description="TODO")
    wwm18: int = Field(1, description="TODO")
    TPPD: str = Field("T", description="TODO")
    wwm19: int = Field(0, description="TODO")
    UBOT: str = Field("F", description="TODO")
    wwm20: int = Field(0, description="TODO")
    ORBITAL: str = Field("F", description="TODO")
    wwm21: int = Field(0, description="TODO")
    wwm22: int = Field(0, description="TODO")
    wwm23: int = Field(0, description="TODO")
    wwm24: int = Field(0, description="TODO")
    wwm25: int = Field(0, description="TODO")
    wwm26: int = Field(0, description="TODO")
    wwm27: int = Field(0, description="TODO")
    wwm28: int = Field(0, description="TODO")
    wwm29: int = Field(0, description="TODO")
    wwm30: int = Field(0, description="TODO")
    wwm31YN: str = Field("!", description="TODO")
    wwm31: int = Field(0, description="TODO")
    wwm32YN: str = Field("!", description="TODO")
    wwm32: int = Field(0, description="TODO")
    wwm33YN: str = Field("!", description="TODO")
    wwm33: int = Field(0, description="TODO")
    wwm34YN: str = Field("!", description="TODO")
    wwm34: int = Field(0, description="TODO")
    wwm35YN: str = Field("!", description="TODO")
    wwm35: int = Field(0, description="TODO")
    wwm36YN: str = Field("!", description="TODO")
    wwm36: int = Field(0, description="TODO")
    wwm37YN: str = Field("!", description="TODO")
    wwm37: int = Field(0, description="TODO")
    iouts: int = Field(7, description="TODO")
    nouts: str = Field(
        "'AWAC_in','AWAC_mid','AWAC_off','SPOT_1002','SPOT_1011','SPOT_1018','SPOT_1026'",
        description="TODO",
    )
    xouts: str = Field(
        "115.6208687,115.5941886,115.58077,115.5942931,115.5830497,115.5807825,115.5960683",
        description="TODO",
    )
    youts: str = Field(
        "-32.611605,-32.611605,-32.613682,-32.6253914,-32.6135870,-32.6294226,-32.6096741",
        description="TODO",
    )
    lsp2d: str = Field("T", description="TODO")
    ac: str = Field("T", description="TODO")
    template: Optional[str] = Field(
        description="The path to the model template",
        default=CSIRO_TEMPLATE,
    )

    # validator example - ensure the following
    # Bottom friction.
    #           nchi=0: drag coefficients specified in drag.gr3; nchi=-1: Manning's
    #           formulation (even for 3D prisms) with n specified in manning.gr3.
    #           nchi=1: bottom roughness (in meters) specified in rough.gr3 (and in this case, negative
    #           or 0 depths in rough.gr3 indicate time-independent Cd, not roughness!).
    #           Cd is calculated using the log law, when dzb>=dzb_min; when dzb<dzb_min,
    #           Cd=Cdmax*exp[dzb_decay*(1-dzb/dzb_min)], where Cdmax=Cd(dzb=dzb_min),
    #           and dzb_decay (<=0) is a decay const specified below. We recommend dzb_decay=0
    #           and may remove this constant in the future.
    #           If iwbl/=0, nchi must =1.
    #   nchi = -1
    #   dzb_min = 0.5 !needed if nchi=1; min. bottom boundary layer thickness [m].
    #   dzb_decay = 0. !needed if nchi=1; a decay const. [-]. should =0
    #   hmin_man = 1.0 !needed if nchi=-1: min. depth in Manning's formulation [m]

    @model_validator(mode="after")
    def validate_bottom_friction(cls, v):
        if v.nchi == 0:
            if v.grid.drag is None:
                raise ValueError("drag.gr3 must be specified when nchi=0")
        elif v.nchi == -1:
            if v.grid.manning is None:
                raise ValueError("manning.gr3 must be specified when nchi=-1")
        elif v.nchi == 1:
            if v.grid.rough is None:
                raise ValueError("rough.gr3 must be specified when nchi=1")
        else:
            raise ValueError("nchi must be 0, -1, or 1")
        return v

    def __call__(self, runtime) -> str:
        # Copy grid files
        self.grid.get(runtime.staging_dir)
        ret = self.model_dump()
        ret.update(self.inputs.get(runtime.staging_dir))
        return ret
