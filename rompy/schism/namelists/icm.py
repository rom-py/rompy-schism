# This file was auto generated from a SCHISM namelist file on 2024-09-13.

from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Marco(NamelistBaseModel):
    nsub: Optional[int] = Field(1, description="Number of subcycles in ICM kinetics")
    iKe: Optional[int] = Field(0, description="")
    Ke0: Optional[float] = Field(
        0.26, description="backgroud light extinction coefficient (1/m)"
    )
    KeC: Optional[float] = Field(0.017, description="Light attenu. due to chlorophyll")
    KeS: Optional[float] = Field(0.07, description="Light attenu. due to TSS")
    KeSalt: Optional[float] = Field(
        -0.02, description="Light attenu. due to CDOM (related to salinity)"
    )
    tss2c: Optional[float] = Field(6.0, description="TSS to carbon ratio")
    iLight: Optional[int] = Field(0, description="")
    alpha: Optional[list] = Field(
        [8.0, 8.0, 8.0],
        description="Initial slope of P-I curve (g[C]*m2/g[Chl]/E) for each phytoplankton group",
    )
    iPR: Optional[int] = Field(1, description="(0: linear formulation; 1: quadratic)")
    PRR: Optional[list] = Field(
        [0.1, 0.2, 0.05],
        description="predation rate by higher trophic level (day-1, or day-1.g-1.m3)",
    )
    wqc0: Optional[list] = Field(
        [
            1.0,
            0.5,
            0.05,
            1.0,
            0.5,
            0.5,
            0.15,
            0.15,
            0.05,
            0.01,
            0.05,
            0.005,
            0.005,
            0.01,
            0.05,
            0.0,
            12.0,
        ],
        description="Initial values for ICM state variables",
    )
    WSP: Optional[list] = Field(
        [
            0.3,
            0.1,
            0.0,
            0.25,
            0.25,
            0.0,
            0.25,
            0.25,
            0.0,
            0.0,
            0.0,
            0.25,
            0.25,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        description="",
    )
    WSPn: Optional[list] = Field(
        [
            0.3,
            0.1,
            0.0,
            0.25,
            0.25,
            0.0,
            0.25,
            0.25,
            0.0,
            0.0,
            0.0,
            0.25,
            0.25,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        description="",
    )
    iSilica: Optional[int] = Field(0, description="")
    iZB: Optional[int] = Field(0, description="")
    iPh: Optional[int] = Field(0, description="")
    iCBP: Optional[int] = Field(0, description="")
    isav_icm: Optional[int] = Field(
        0, description="Submerged Aquatic Vegetation switch. 0: OFF, 1: ON"
    )
    iveg_icm: Optional[int] = Field(
        0, description="Intertidal vegetation switch. 0: OFF, 1: ON"
    )
    iSed: Optional[int] = Field(1, description="")
    iBA: Optional[int] = Field(0, description="")
    iRad: Optional[int] = Field(0, description="")
    isflux: Optional[int] = Field(
        0,
        description="Atmospheric fluxes option. 0: OFF, 1: additional nutrient fluxes from ICM_sflux.th.nc",
    )
    ibflux: Optional[int] = Field(
        0,
        description="Bottom fluxes option. 0: OFF, 1: additional nutrient fluxes from ICM_bflux.th.nc",
    )
    iout_icm: Optional[int] = Field(
        0,
        description="ICM station outputs switch. 0: OFF, 1: ON (requires istation.in with *.bp format)",
    )
    nspool_icm: Optional[int] = Field(
        24, description="Output frequency for ICM station outputs"
    )
    iLimit: Optional[int] = Field(0, description="")
    idry_icm: Optional[int] = Field(
        0,
        description="Shallow kinetic biochemical process option. 0: jump dry elements, keep last wet value, 1: turn on shallow kinetic biochemical process",
    )

    @field_validator("nsub")
    @classmethod
    def validate_nsub(cls, v):
        if v < 1:
            raise ValueError("nsub must be a positive integer")
        return v

    @field_validator("tss2c")
    @classmethod
    def validate_tss2c(cls, v):
        if v <= 0:
            raise ValueError("tss2c must be positive")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v):
        if len(v) != 3 or any(x <= 0 for x in v):
            raise ValueError("alpha must be a list of 3 positive values")
        return v

    @field_validator("wqc0")
    @classmethod
    def validate_wqc0(cls, v):
        if len(v) != 16 or any(x < 0 for x in v):
            raise ValueError("wqc0 must be a list of 16 non-negative values")
        return v

    @field_validator("isav_icm")
    @classmethod
    def validate_isav_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("isav_icm must be 0 or 1")
        return v

    @field_validator("iveg_icm")
    @classmethod
    def validate_iveg_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("iveg_icm must be 0 or 1")
        return v

    @field_validator("isflux")
    @classmethod
    def validate_isflux(cls, v):
        if v not in [0, 1]:
            raise ValueError("isflux must be 0 or 1")
        return v

    @field_validator("ibflux")
    @classmethod
    def validate_ibflux(cls, v):
        if v not in [0, 1]:
            raise ValueError("ibflux must be 0 or 1")
        return v

    @field_validator("iout_icm")
    @classmethod
    def validate_iout_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("iout_icm must be 0 or 1")
        return v

    @field_validator("nspool_icm")
    @classmethod
    def validate_nspool_icm(cls, v):
        if v <= 0:
            raise ValueError("nspool_icm must be positive")
        return v

    @field_validator("idry_icm")
    @classmethod
    def validate_idry_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("idry_icm must be 0 or 1")
        return v


class Core(NamelistBaseModel):
    GPM: Optional[list] = Field([2.5, 2.8, 3.5], description="PB growth rates (day-1)")
    TGP: Optional[list] = Field(
        [15.0, 22.0, 27.0], description="optimal temp. for PB growth (oC)"
    )
    KTGP: Optional[list] = Field(
        [0.005, 0.004, 0.003, 0.008, 0.006, 0.004],
        description="temp. dependence for PB growth; dim(PB=1:3,1:2) (oC-2)",
    )
    MTR: Optional[list] = Field(
        [0.0, 0.0, 0.0], description="PB photorespiration coefficient (0<MTR<1)"
    )
    MTB: Optional[list] = Field(
        [0.01, 0.02, 0.03], description="PB metabolism rates (day-1)"
    )
    TMT: Optional[list] = Field(
        [20.0, 20.0, 20.0], description="reference temp. for PB metabolism (oC)"
    )
    KTMT: Optional[list] = Field(
        [0.0322, 0.0322, 0.0322],
        description="temp. dependence for PB metabolism (oC-1)",
    )
    FCP: Optional[list] = Field(
        [0.35, 0.3, 0.2, 0.55, 0.5, 0.5, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        description="fractions of PB carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,3)",
    )
    FNP: Optional[list] = Field(
        [
            0.35,
            0.35,
            0.35,
            0.5,
            0.5,
            0.5,
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.0,
            0.0,
            0.0,
        ],
        description="fractions of PB nitrogen into (RPON,LPON,DON,NH4,SRPON)",
    )
    FPP: Optional[list] = Field(
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
        description="fractions of PB Phosphorus into (RPOP,LPOP,DOP,PO4,SRPOP)",
    )
    FCM: Optional[list] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        description="fractions of PB metabolism carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,4)",
    )
    FNM: Optional[list] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="fractions of PB metabolism N. into (RPON,LPON,DON,NH4,SRPON); dim(PB=1:3,5)",
    )
    FPM: Optional[list] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="fractions of PB metabolism P. into (RPOP,LPOP,DOP,PO4,SRPOP); dim(PB=1:3,5)",
    )
    Nit: Optional[float] = Field(0.07, description="maximum nitrification rate (day-1)")
    TNit: Optional[float] = Field(
        27.0, description="optimal temp. for nitrification (oC)"
    )
    KTNit: Optional[list] = Field(
        [0.0045, 0.0045],
        description="temp. dependence (T<=TNit & T>TNit) for nitrification (oC-2)",
    )
    KhDOn: Optional[float] = Field(
        1.0, description="DO half saturation for nitrification (mg/L)"
    )
    KhDOox: Optional[float] = Field(
        0.5,
        description="DO half saturation for dentrification & DOC's oxic respiration (mg/L)",
    )
    KhNO3dn: Optional[float] = Field(
        0.1, description="NO3 half saturation for denitrification (mg/L)"
    )
    KC0: Optional[list] = Field(
        [0.005, 0.075, 0.2], description="minimum decay rate of RPOC,LPOC,DOC (day-1)"
    )
    KN0: Optional[list] = Field(
        [0.005, 0.075, 0.2], description="minimum decay rate of RPON,LPON,DON (day-1)"
    )
    KP0: Optional[list] = Field(
        [0.005, 0.075, 0.2], description="minimum decay rate of RPOP,LPOP,DOP (day-1)"
    )
    KCalg: Optional[list] = Field(
        [0.0, 0.0, 0.0],
        description="algae effect on RPOC,LPOC,DOC decay (day-1.m3.g[C]-1)",
    )
    KNalg: Optional[list] = Field(
        [0.0, 0.0, 0.0],
        description="algae effect on RPON,LPON,DON decay (day-1.m3.g[C]-1)",
    )
    KPalg: Optional[list] = Field(
        [0.0, 0.0, 0.0],
        description="algae effect on RPOP,LPOP,DOP decay (day-1.m3.g[C]-1)",
    )
    TRM: Optional[list] = Field(
        [20.0, 20.0, 20.0], description="reference temp. for (RPOM,LPOM,DOM) decay (oC)"
    )
    KTRM: Optional[list] = Field(
        [0.069, 0.069, 0.069],
        description="temp. dependence for (RPOM,LPOM,DOM) decay (oC-1)",
    )
    KSR0: Optional[list] = Field(
        [0.001, 0.001, 0.001], description="decay rates of SRPOC,SRPON,SRPOP (day-1)"
    )
    TRSR: Optional[list] = Field(
        [20.0, 20.0, 20.0],
        description="reference temp. for (SRPOC,SRPON,SRPOP) decay (oC)",
    )
    KTRSR: Optional[list] = Field(
        [0.069, 0.069, 0.069],
        description="temp. dependence for (SRPOC,SRPON,SRPOP) decay (oC-1)",
    )
    KPIP: Optional[float] = Field(0.0, description="dissolution rate of PIP (day-1)")
    KCD: Optional[float] = Field(
        1.0, description="oxidation rate of COD at TRCOD (day-1)"
    )
    TRCOD: Optional[float] = Field(
        20.0, description="reference temp. for COD oxidation (oC)"
    )
    KTRCOD: Optional[float] = Field(
        0.041, description="temp. dependence for COD oxidation (oC-1)"
    )
    KhCOD: Optional[float] = Field(
        1.5, description="COD half saturation for COD oxidation (mg[O2]/L)"
    )
    KhN: Optional[list] = Field(
        [0.01, 0.01, 0.01], description="nitrogen half saturation (mg/L)"
    )
    KhP: Optional[list] = Field(
        [0.001, 0.001, 0.001], description="phosphorus half saturation (mg/L)"
    )
    KhSal: Optional[list] = Field(
        ["1e6", "1e6", 0.1],
        description="salinity when PB growth is halved (PSU); (1e6: no salinity stress)",
    )
    c2chl: Optional[list] = Field(
        [0.059, 0.059, 0.059],
        description="Carbon to chlorophyll ratio for three phytoplankton species (g[C]/mg[Chl])",
    )
    n2c: Optional[list] = Field(
        [0.167, 0.167, 0.167],
        description="Nitrogen to carbon ratio for three phytoplankton species",
    )
    p2c: Optional[list] = Field(
        [0.02, 0.02, 0.02],
        description="Phosphorus to carbon ratio for three phytoplankton species",
    )
    o2c: Optional[float] = Field(
        2.67, description="Oxygen to carbon ratio in respiration"
    )
    o2n: Optional[float] = Field(
        4.33, description="Oxygen to ammonium ratio (g[O2]/g[NH4])"
    )
    dn2c: Optional[float] = Field(
        0.933,
        description="Mass of NO3 consumed per mass of DOC oxidized in denitrification (g[N]/g[C])",
    )
    an2c: Optional[float] = Field(
        0.5, description="Ratio of denitrification rate to oxic DOC respiration rate"
    )
    KhDO: Optional[list] = Field(
        [0.5, 0.5, 0.5], description="DO half saturation for PB's DOC excretion (mg/L)"
    )
    KPO4p: Optional[float] = Field(
        0.0, description="coefficient relating PO4 sorption to TSS"
    )
    WRea: Optional[float] = Field(
        0.0, description="baseline wind-induced reaeration coefficient for DO (day-1)"
    )
    PBmin: Optional[list] = Field(
        [0.01, 0.01, 0.01], description="minimum PB concentration (mg[C]/L)"
    )
    dz_flux: Optional[list] = Field(
        [1.0, 1.0],
        description="Surface/bottom thickness (m) within which surface flux/bottom flux are redistributed",
    )

    @field_validator("c2chl")
    @classmethod
    def validate_c2chl(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("c2chl must be a list of 3 positive numbers")
        return v

    @field_validator("n2c")
    @classmethod
    def validate_n2c(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v)
        ):
            raise ValueError("n2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("p2c")
    @classmethod
    def validate_p2c(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v)
        ):
            raise ValueError("p2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("o2c")
    @classmethod
    def validate_o2c(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("o2c must be a positive number")
        return v

    @field_validator("o2n")
    @classmethod
    def validate_o2n(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("o2n must be a positive number")
        return v

    @field_validator("dn2c")
    @classmethod
    def validate_dn2c(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("dn2c must be a positive number")
        return v

    @field_validator("an2c")
    @classmethod
    def validate_an2c(cls, v):
        if not isinstance(v, (int, float)) or v <= 0 or v > 1:
            raise ValueError("an2c must be a number between 0 and 1")
        return v

    @field_validator("dz_flux")
    @classmethod
    def validate_dz_flux(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 2
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("dz_flux must be a list of 2 positive numbers")
        return v

    @model_validator(mode="after")
    def validate_fcp_sum(self):
        for i in range(3):
            if sum(self.fcp[i::3]) != 1:
                raise ValueError(f"Sum of FCP fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fnp_sum(self):
        for i in range(3):
            if sum(self.fnp[i::3]) != 1:
                raise ValueError(f"Sum of FNP fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fpp_sum(self):
        for i in range(3):
            if sum(self.fpp[i::3]) != 1:
                raise ValueError(f"Sum of FPP fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fcm_sum(self):
        for i in range(3):
            if sum(self.fcm[i::3]) != 0.1:
                raise ValueError(
                    f"Sum of FCM fractions for species {i+1} must equal 0.1"
                )
        return self

    @model_validator(mode="after")
    def validate_fnm_sum(self):
        for i in range(5):
            if sum(self.fnm[i::3]) != 1:
                raise ValueError(f"Sum of FNM fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fpm_sum(self):
        for i in range(5):
            if sum(self.fpm[i::3]) != 1:
                raise ValueError(f"Sum of FPM fractions for species {i+1} must equal 1")
        return self


class Sfm(NamelistBaseModel):
    btemp0: Optional[float] = Field(
        5.0, description="Initial temperature of the sediment layer in degrees Celsius"
    )
    bstc0: Optional[float] = Field(
        0.1, description="Initial surface transfer coefficient for the sediment layer"
    )
    bSTR0: Optional[float] = Field(0.0, description="benthic stress (day)")
    bThp0: Optional[float] = Field(0.0, description="consective days of hypoxia (day)")
    bTox0: Optional[float] = Field(
        0.0, description="consective days of oxic condition after hypoxia event (day)"
    )
    bNH40: Optional[float] = Field(4.0, description="NH4")
    bNO30: Optional[float] = Field(1.0, description="NO3")
    bPO40: Optional[float] = Field(5.0, description="PO4")
    bH2S0: Optional[float] = Field(250.0, description="H2S")
    bCH40: Optional[float] = Field(40.0, description="CH4")
    bPOS0: Optional[float] = Field(500.0, description="POS")
    bSA0: Optional[float] = Field(500.0, description="SA")
    bPOC0: Optional[list] = Field([1000.0, 3000.0, 5000.0], description="POC(G=1:3)")
    bPON0: Optional[list] = Field([150.0, 500.0, 1500.0], description="PON(G=1:3)")
    bPOP0: Optional[list] = Field([30.0, 300.0, 500.0], description="POP(G=1:3)")
    bdz: Optional[float] = Field(0.1, description="Sediment thickness (m)")
    bVb: Optional[str] = Field(
        "1.37e-5", description="burial rate (m.day-1); 1 cm/yr=2.74e-5 m.day-1"
    )
    bsolid: Optional[list] = Field(
        [0.5, 0.5],
        description="Sediment solid concentrations in Layer 1 and Layer 2 (Kg/L)",
    )
    bdiff: Optional[str] = Field(
        "1.8e-7", description="Diffusion coefficient for sediment temperature (m2/s)"
    )
    bTR: Optional[int] = Field(20, description="reference temp. for sediment processes")
    bVpmin: Optional[str] = Field(
        "3.0e-6", description="minimum particle mixing velocity coefficient (m.day-1)"
    )
    bVp: Optional[str] = Field(
        "1.2e-4", description="particle mixing velocity coefficient (m.day-1)"
    )
    bVd: Optional[str] = Field(
        "1.0e-3", description="diffusion velocity coefficient (m.day-1)"
    )
    bKTVp: Optional[float] = Field(
        1.117, description="temp. dependece of particle mixing velocity"
    )
    bKTVd: Optional[float] = Field(
        1.08, description="temp. dependece of diffusion velocity"
    )
    bKST: Optional[float] = Field(
        0.03, description="1st order decay rate of benthic stress  (day-1)"
    )
    bSTmax: Optional[float] = Field(
        20.0,
        description="maximum value of benthic stress (day) (note: smaller than 1/bKST)",
    )
    bKhDO_Vp: Optional[float] = Field(
        4.0, description="DO half-saturation of particle mixing (mg/L)"
    )
    bDOc_ST: Optional[float] = Field(
        1.0, description="DO criteria for benthic stress (mg/L)"
    )
    banoxic: Optional[float] = Field(
        10.0,
        description="consective days of hypoxia causing maximum benthic stress (day)",
    )
    boxic: Optional[float] = Field(
        45.0, description="time lag for bethos recovery from hypoxia event (day)"
    )
    bp2d: Optional[float] = Field(
        0.0,
        description="ratio from mixing coef. to diffusion coef. (benthos enhanced effect)",
    )
    bKC: Optional[list] = Field(
        [0.035, 0.0018, 0.0], description="decay rate of POC (3G class) at bTR (day-1)"
    )
    bKN: Optional[list] = Field(
        [0.035, 0.0018, 0.0], description="decay rate of PON (3G class) at bTR (day-1)"
    )
    bKP: Optional[list] = Field(
        [0.035, 0.0018, 0.0], description="decay rate of POP (3G class) at bTR (day-1)"
    )
    bKTC: Optional[list] = Field(
        [1.1, 1.15, 1.17], description="temp. dependence of POC decay (oC-1)"
    )
    bKTN: Optional[list] = Field(
        [1.1, 1.15, 1.17], description="temp. dependence of PON decay (oC-1)"
    )
    bKTP: Optional[list] = Field(
        [1.1, 1.15, 1.17], description="temp. dependence of POP decay (oC-1)"
    )
    bFCP: Optional[list] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto POC into sed POC (G3,PB=1:3)",
    )
    bFNP: Optional[list] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto PON into sed PON (G3,PB=1:3)",
    )
    bFPP: Optional[list] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto POP into sed POP (G3,PB=1:3)",
    )
    bFCM: Optional[list] = Field(
        [0.0, 0.43, 0.57], description="refractory POC into sed POC(3G)"
    )
    bFNM: Optional[list] = Field(
        [0.0, 0.54, 0.46], description="refractory PON into sed PON(3G)"
    )
    bFPM: Optional[list] = Field(
        [0.0, 0.43, 0.57], description="refractory POP into sed POP(3G)"
    )
    bKNH4f: Optional[float] = Field(
        0.2, description="NH4 reaction rate in freshwater  at bTR (1st layer) (m/day)"
    )
    bKNH4s: Optional[float] = Field(
        0.14, description="NH4 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bKTNH4: Optional[float] = Field(
        1.08, description="temp. dependency for NH4 reaction (oC-1)"
    )
    bKhNH4: Optional[float] = Field(
        1.5, description="half-stauration NH4 for nitrification (g/m3)"
    )
    bKhDO_NH4: Optional[float] = Field(
        2.0, description="half-stauration DO for nitrification (g/m3)"
    )
    bpieNH4: Optional[float] = Field(
        1.0, description="partition coefficients of NH4 in Layer 1 & 2 (Kg-1.L)"
    )
    bsaltn: Optional[float] = Field(
        1.0,
        description="salinity criteria of fresh/salty water for NH4/NO3 reaction (PSU)",
    )
    bKNO3f: Optional[float] = Field(
        0.3, description="NO3 reaction rate in freshwater at bTR (1st layer) (m/day)"
    )
    bKNO3s: Optional[float] = Field(
        0.125, description="NO3 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bKNO3: Optional[float] = Field(
        0.25, description="NO3 reaction rate (2nd layer) (m/day)"
    )
    bKTNO3: Optional[float] = Field(
        1.08, description="temp. dependency for NO3 reaction (oC-1)"
    )
    bKH2Sd: Optional[float] = Field(
        0.2, description="dissolved H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bKH2Sp: Optional[float] = Field(
        0.4, description="particulate H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bKTH2S: Optional[float] = Field(
        1.08, description="temp. dependency for H2S reaction (oC-1)"
    )
    bpieH2Ss: Optional[float] = Field(
        100.0, description="partition coefficient of NH4 in Layer 1 (Kg-1.L)"
    )
    bpieH2Sb: Optional[float] = Field(
        100.0, description="partition coefficient of NH4 in Layer 2 (Kg-1.L)"
    )
    bKhDO_H2S: Optional[float] = Field(
        8.0, description="O2 constant to normalize H2S oxidation (g[O2]/m3)"
    )
    bsaltc: Optional[float] = Field(
        1.0,
        description="salinity criteria of fresh/salty water for carbon reaction (PSU)",
    )
    bKCH4: Optional[float] = Field(
        0.2, description="CH4 reaction rate at bTR (1st layer) (m/day)"
    )
    bKTCH4: Optional[float] = Field(
        1.08, description="temp. dependency for CH4 reaction"
    )
    bKhDO_CH4: Optional[float] = Field(
        0.2, description="half-saturation DO for CH4 oxidation (g[O2]/m3)"
    )
    bo2n: Optional[float] = Field(
        2.86, description="oxygen to nitrogen ratio in sediment (denitrification)"
    )
    bpiePO4: Optional[float] = Field(
        50.0, description="partition coefficient of PO4 in Layer 2 (Kg-1.L)"
    )
    bKOPO4f: Optional[float] = Field(
        3000.0,
        description="oxygen dependency for PO4 sorption in freshwater in Layer 1",
    )
    bKOPO4s: Optional[float] = Field(
        300.0,
        description="oxygen dependency for PO4 sorption in salty water in Layer 1",
    )
    bDOc_PO4: Optional[float] = Field(
        1.0, description="DO criteria for PO4 sorptiona (g[O2]/m3)"
    )
    bsaltp: Optional[float] = Field(
        1.0,
        description="salinity criteria of fresh/salty water for PO4 partition (PSU)",
    )
    bKS: Optional[float] = Field(0.5, description="decay rate of POS (3G class) at bTR")
    bKTS: Optional[float] = Field(1.1, description="temp. dependence of POS decay")
    bSIsat: Optional[float] = Field(
        40.0, description="silica saturation conc. in pore water (g[Si]/m3)"
    )
    bpieSI: Optional[float] = Field(
        100.0, description="partition coefficient of silica in Layer 2 (Kg-1.L)"
    )
    bKOSI: Optional[float] = Field(
        10.0, description="oxygen dependency for silica sorption in Layer 1"
    )
    bKhPOS: Optional[str] = Field(
        "5.0e4", description="POS half saturation for POS dissolution (g/m3)"
    )
    bDOc_SI: Optional[float] = Field(
        1.0, description="DO criteria for silica sorptiona (g[O2]/m3)"
    )
    bJPOSa: Optional[float] = Field(
        0.0,
        description="additional POS flux associated with POM detrius beside algea (g.m-2.day-1)",
    )
    bFCs: Optional[list] = Field(
        [0.65, 0.255, 0.095], description="SAV POC into 3G sed 3G POC"
    )
    bFNs: Optional[list] = Field(
        [0.65, 0.3, 0.05], description="SAV PON into 3G sed 3G PON"
    )
    bFPs: Optional[list] = Field(
        [0.65, 0.255, 0.095], description="SAV POP into 3G sed 3G POP"
    )
    bFCv: Optional[list] = Field(
        [0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095],
        description="VEG POC into sed POC (G3,PB=1:3)",
    )
    bFNv: Optional[list] = Field(
        [0.65, 0.3, 0.05, 0.65, 0.3, 0.05, 0.65, 0.3, 0.05],
        description="VEG PON into sed PON (G3,PB=1:3)",
    )
    bFPv: Optional[list] = Field(
        [0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095],
        description="VEG POP into sed POP (G3,PB=1:3)",
    )

    @field_validator("btemp0")
    @classmethod
    def validate_btemp0(cls, v):
        if v < -2 or v > 40:
            raise ValueError("btemp0 must be between -2 and 40")
        return v

    @field_validator("bstc0")
    @classmethod
    def validate_bstc0(cls, v):
        if v < 0:
            raise ValueError("bstc0 must be non-negative")
        return v

    @field_validator("bdz")
    @classmethod
    def validate_bdz(cls, v):
        if v <= 0:
            raise ValueError("bdz must be positive")
        return v

    @field_validator("bsolid")
    @classmethod
    def validate_bsolid(cls, v):
        if len(v) != 2 or any(x <= 0 or x > 1 for x in v):
            raise ValueError("bsolid must be a list of 2 values between 0 and 1")
        return v

    @field_validator("bdiff")
    @classmethod
    def validate_bdiff(cls, v):
        if v <= 0:
            raise ValueError("bdiff must be positive")
        return v


class Silica(NamelistBaseModel):
    FSP: Optional[list] = Field(
        [0.9, 0.1], description="fractions of diatom silica into (SU,SA)"
    )
    FSM: Optional[list] = Field(
        [0.5, 0.5], description="fractions of diatom metabolism Si into (SU,SA)"
    )
    KS: Optional[float] = Field(
        0.03, description="dissolution rate of SU at TRS (day-1)"
    )
    TRS: Optional[float] = Field(
        20.0, description="reference temp. for SU dissolution (oC)"
    )
    KTRS: Optional[float] = Field(
        0.092, description="temp. dependence for SU dissolution (oC-1)"
    )
    KhS: Optional[list] = Field(
        [0.05, 0.0, 0.0],
        description="silica half saturation (mg/L); (0.0: no Si limitation)",
    )
    s2c: Optional[list] = Field(
        [0.5, 0.0, 0.0],
        description="silica to carbon ratio for phytolankton; (0.0: no Si uptake)",
    )
    KSAp: Optional[float] = Field(
        0.0, description="coefficient relating Silicate(SA) sorption to TSS"
    )


class Zb(NamelistBaseModel):
    zGPM: Optional[list] = Field(
        [
            0.0,
            0.0,
            1.75,
            1.75,
            1.75,
            1.75,
            1.75,
            1.75,
            1.0,
            0.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ],
        description="ZB predation rate(day-1); dim(prey=1:8, ZB=1:2)",
    )
    zKhG: Optional[list] = Field(
        [
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
        ],
        description="reference prey conc.(mg/L); dim(prey=1:8, ZB=1:2)",
    )
    zTGP: Optional[list] = Field(
        [25.0, 25.0], description="optimal temp. for ZB growth (oC)"
    )
    zKTGP: Optional[list] = Field(
        [0.0035, 0.008, 0.025, 0.03],
        description="temp. dependence for ZB growth (T<=zTGP & T>zTGP); dim(ZB=1:2,1:2) (oC-2)",
    )
    zAG: Optional[float] = Field(
        0.75, description="ZB assimilation efficiency ratio (0-1)"
    )
    zRG: Optional[float] = Field(
        0.1, description="ZB respiration ratio when it grazes (0-1)"
    )
    zMRT: Optional[list] = Field([0.02, 0.02], description="ZB mortality rates (day-1)")
    zMTB: Optional[list] = Field(
        [0.254, 0.186], description="ZB metabolism rates (day-1)"
    )
    zTMT: Optional[list] = Field(
        [20.0, 20.0], description="reference temp. for ZB metabolism (oC)"
    )
    zKTMT: Optional[list] = Field(
        [0.0693, 0.0693], description="temp. dependence for ZB metabolism (oC-1)"
    )
    zFCP: Optional[list] = Field(
        [0.35, 0.55, 0.1], description="fractions of ZB carbon into (RPOC,LPOC,DOC)"
    )
    zFNP: Optional[list] = Field(
        [0.35, 0.5, 0.1, 0.05],
        description="fractions of ZB nitrogen into (RPON,LPON,DON,NH4)",
    )
    zFPP: Optional[list] = Field(
        [0.1, 0.2, 0.5, 0.2],
        description="fractions of ZB Phosphorus into (RPOP,LPOP,DOP,PO4)",
    )
    zFSP: Optional[list] = Field(
        [0.7, 0.25], description="fractions of ZB silica into (SU,SA)"
    )
    zFCM: Optional[list] = Field(
        [0.1, 0.0],
        description="fractions of ZB metabolism carbon into DOC; dim(ZB=1:2)",
    )
    zFNM: Optional[list] = Field(
        [0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1],
        description="fractions of ZB metabolism N. into (RPON,LPON,DON,NH4); dim(ZB=1:2,4)",
    )
    zFPM: Optional[list] = Field(
        [0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1],
        description="fractions of ZB metabolism P. into (RPOP,LPOP,DOP,PO4); dim(ZB=1:2,4)",
    )
    zFSM: Optional[list] = Field(
        [0.5, 0.4, 0.5, 0.6],
        description="fractions of ZB metabolism Si into (SU,SA); dim(ZB=1:2,2)",
    )
    zKhDO: Optional[list] = Field(
        [0.5, 0.5], description="DO half saturation for ZB's DOC excretion (mg/L)"
    )
    zn2c: Optional[list] = Field(
        [0.2, 0.2], description="Nitrogen to carbon ratio for zooplankton"
    )
    zp2c: Optional[list] = Field(
        [0.02, 0.02], description="Phosphorus to carbon ratio for zooplankton"
    )
    zs2c: Optional[list] = Field(
        [0.5, 0.5], description="Silica to carbon ratio for zooplankton"
    )
    z2pr: Optional[list] = Field(
        [0.5, 0.5],
        description="Ratio converting zooplankton and phytoplankton biomass to predation rates on zooplankton (L.mg^-1.day^-1)",
    )
    p2pr: Optional[float] = Field(
        0.25,
        description="Ratio converting zooplankton and phytoplankton biomass to predation rates on phytoplankton (L.mg^-1.day^-1)",
    )

    @field_validator("zn2c")
    @classmethod
    def validate_zn2c(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zn2c must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zp2c")
    @classmethod
    def validate_zp2c(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zp2c must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zs2c")
    @classmethod
    def validate_zs2c(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zs2c must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("z2pr")
    @classmethod
    def validate_z2pr(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("z2pr must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("p2pr")
    @classmethod
    def validate_p2pr(cls, v):
        return float(v)


class Ph_icm(NamelistBaseModel):
    ppatch0: Optional[int] = Field(
        -999,
        description="Region flag for pH modeling. If set to 1, pH modeling is enabled for all elements. If set to -999, spatial pH modeling is used.",
    )
    pKCACO3: Optional[float] = Field(
        60.0, description="dissulution bewteen CaCO3 and Ca++"
    )
    pKCA: Optional[float] = Field(
        60.0, description="sediment surface transfer coefficient from CaCO3 to Ca++"
    )
    pRea: Optional[float] = Field(1.0, description="reaeration rate for CO2")
    inu_ph: Optional[int] = Field(
        0,
        description="Nudge option for pH model. Controls whether pH values are adjusted during simulation.",
    )

    @field_validator("ppatch0")
    @classmethod
    def validate_ppatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("ppatch0 must be either 1 or -999")
        return v

    @field_validator("inu_ph")
    @classmethod
    def validate_inu_ph(cls, v):
        if v not in [0, 1]:
            raise ValueError("inu_ph must be either 0 or 1")
        return v


class Sav(NamelistBaseModel):
    spatch0: Optional[int] = Field(
        -999,
        description="Region flag for SAV. 1 indicates SAV is active on all elements, -999 indicates spatial distribution",
    )
    stleaf0: Optional[int] = Field(
        -999, description="Initial concentration of total SAV leaf biomass"
    )
    ststem0: Optional[int] = Field(
        -999, description="Initial concentration of total SAV stem biomass"
    )
    stroot0: Optional[int] = Field(
        -999, description="Initial concentration of total SAV root biomass"
    )
    sGPM: Optional[float] = Field(0.1, description="maximum growth rate (day-1)")
    sTGP: Optional[int] = Field(32, description="optimal growth temperature (oC)")
    sKTGP: Optional[list] = Field(
        [0.003, 0.005], description="temp. dependence for growth (T<=sTGP & T>sTGP)"
    )
    sFAM: Optional[float] = Field(
        0.2, description="fraction of leaf production to active metabolism"
    )
    sFCP: Optional[list] = Field(
        [0.6, 0.3, 0.1], description="fractions of production to leaf/stem/root biomass"
    )
    sMTB: Optional[list] = Field(
        [0.02, 0.02, 0.02], description="metabolism rates of leaf/stem/root"
    )
    sTMT: Optional[list] = Field(
        [20, 20, 20], description="reference temp. for leaf/stem/root metabolism"
    )
    sKTMT: Optional[list] = Field(
        [0.069, 0.069, 0.069],
        description="temp. dependence of leaf/stem/root metabolism",
    )
    sFCM: Optional[list] = Field(
        [0.05, 0.15, 0.3, 0.5],
        description="fractions of metabolism C into (RPOC,LPOC,DOC,CO2)",
    )
    sFNM: Optional[list] = Field(
        [0.05, 0.15, 0.3, 0.5],
        description="fractions of metabolism N into (RPON,LPON,DON,NH4)",
    )
    sFPM: Optional[list] = Field(
        [0.05, 0.1, 0.35, 0.5],
        description="fractions of metabolism P into (RPOP,LPOP,DOP,PO4)",
    )
    sKhNw: Optional[float] = Field(
        0.01, description="nitrogen half saturation in water column"
    )
    sKhNs: Optional[float] = Field(
        0.1, description="nitrogen half saturation in sediments"
    )
    sKhNH4: Optional[float] = Field(0.1, description="ammonium half saturation")
    sKhPw: Optional[float] = Field(
        0.001, description="phosphorus half saturation in water column"
    )
    sKhPs: Optional[float] = Field(
        0.01, description="phosphorus half saturation in sediments"
    )
    salpha: Optional[float] = Field(
        0.006, description="Initial slope of the SAV photosynthesis-irradiance curve"
    )
    sKe: Optional[float] = Field(
        0.045, description="light attenuation due to sav absorption"
    )
    shtm: Optional[list] = Field(
        [0.054, 2.0], description="Minimum (base) and maximum SAV canopy height"
    )
    s2ht: Optional[list] = Field(
        [0.0036, 0.0036, 0.0],
        description="Coefficients for converting SAV leaf, stem, and root biomass to canopy height",
    )
    sc2dw: Optional[float] = Field(
        0.38, description="Carbon to dry weight ratio of SAV"
    )
    s2den: Optional[int] = Field(
        10, description="Coefficient for computing SAV density from leaf biomass"
    )

    @field_validator("spatch0")
    @classmethod
    def validate_spatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("spatch0 must be either 1 or -999")
        return v

    @field_validator("stleaf0")
    @classmethod
    def validate_stleaf0(cls, v):
        if v < 0 and v != -999:
            raise ValueError("stleaf0 must be non-negative or -999")
        return v

    @field_validator("ststem0")
    @classmethod
    def validate_ststem0(cls, v):
        if v < 0 and v != -999:
            raise ValueError("ststem0 must be non-negative or -999")
        return v

    @field_validator("stroot0")
    @classmethod
    def validate_stroot0(cls, v):
        if v < 0 and v != -999:
            raise ValueError("stroot0 must be non-negative or -999")
        return v

    @field_validator("salpha")
    @classmethod
    def validate_salpha(cls, v):
        if v <= 0:
            raise ValueError("salpha must be positive")
        return v

    @field_validator("shtm")
    @classmethod
    def validate_shtm(cls, v):
        if len(v) != 2 or v[0] >= v[1] or any(x <= 0 for x in v):
            raise ValueError(
                "shtm must be a list of two positive numbers with the first less than the second"
            )
        return v

    @field_validator("s2ht")
    @classmethod
    def validate_s2ht(cls, v):
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("s2ht must be a list of three non-negative numbers")
        return v

    @field_validator("sc2dw")
    @classmethod
    def validate_sc2dw(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("sc2dw must be between 0 and 1")
        return v

    @field_validator("s2den")
    @classmethod
    def validate_s2den(cls, v):
        if v <= 0:
            raise ValueError("s2den must be positive")
        return v


class Stem(NamelistBaseModel):
    sn2c: Optional[float] = Field(
        0.09,
        description="Nitrogen to carbon ratio of submerged aquatic vegetation (SAV)",
    )
    sp2c: Optional[float] = Field(0.01, description="Phosphorus to carbon ratio")
    so2c: Optional[float] = Field(2.67, description="Oxygen to carbon ratio")

    @field_validator("sn2c")
    @classmethod
    def validate_sn2c(cls, v):
        if v < 0 or v > 1:
            raise ValueError("sn2c must be between 0 and 1")
        return v

    @field_validator("sp2c")
    @classmethod
    def validate_sp2c(cls, v):
        if v < 0 or v > 1:
            raise ValueError("sp2c must be between 0 and 1")
        return v

    @field_validator("so2c")
    @classmethod
    def validate_so2c(cls, v):
        if v <= 0:
            raise ValueError("so2c must be greater than 0")
        return v


class Veg(NamelistBaseModel):
    vpatch0: Optional[int] = Field(
        -999,
        description="Region flag for VEG. (1: ON for all elements; -999: spatial distribution)",
    )
    vtleaf0: Optional[list] = Field(
        [100.0, 100.0, 100.0],
        description="Initial concentration for total vegetation leaf biomass (3 values for different vegetation types)",
    )
    vtstem0: Optional[list] = Field(
        [100.0, 100.0, 100.0],
        description="Initial concentration for total vegetation stem biomass (3 values for different vegetation types)",
    )
    vtroot0: Optional[list] = Field(
        [30.0, 30.0, 30.0],
        description="Initial concentration for total vegetation root biomass (3 values for different vegetation types)",
    )
    vGPM: Optional[list] = Field(
        [0.1, 0.1, 0.1], description="maximum growth rate (day-1)"
    )
    vFAM: Optional[list] = Field(
        [0.2, 0.2, 0.2], description="fractions of leaf production to active metabolism"
    )
    vTGP: Optional[list] = Field(
        [32.0, 32.0, 32.0], description="optimal growth temperature (oC)"
    )
    vKTGP: Optional[list] = Field(
        [0.003, 0.003, 0.003, 0.005, 0.005, 0.005],
        description="temp. dependence(3,2) for growth (T<=vTGP & T>vTGP)",
    )
    vFCP: Optional[list] = Field(
        [0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1],
        description="fractions of production to leaf/stem/root biomass; dim(veg,leaf/stem/root)",
    )
    vMTB: Optional[list] = Field(
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01],
        description="metabolism rates of leaf/stem/root; dim(veg,leaf/stem/root)",
    )
    vTMT: Optional[list] = Field(
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        description="reference temp. for leaf/stem/root metabolism",
    )
    vKTMT: Optional[list] = Field(
        [0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069],
        description="temp. depdenpence for leaf/stem/root metabolism",
    )
    vFNM: Optional[list] = Field(
        [0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
        description="fractions(3,4) of metabolism N into (RPON,LPON,DON,NH4)",
    )
    vFPM: Optional[list] = Field(
        [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5],
        description="fractions(3,4) of metabolism P into (RPOP,LPOP,DOP,PO4)",
    )
    vFCM: Optional[list] = Field(
        [0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
        description="fractions(3,4) of metabolism N into (RPOC,LPOC,DOC,CO2)",
    )
    ivNc: Optional[int] = Field(
        1, description="recycled veg N goes to (0: sediment; 1: water)"
    )
    ivPc: Optional[int] = Field(
        1, description="recycled veg P goes to (0: sediment; 1: water)"
    )
    vKhNs: Optional[list] = Field(
        [0.1, 0.1, 0.1], description="nitrogen half saturation in sediments"
    )
    vKhPs: Optional[list] = Field(
        [0.01, 0.01, 0.01], description="phosphorus half saturation in sediments"
    )
    vScr: Optional[list] = Field(
        [35.0, 35.0, 35.0], description="reference sality for computing veg growth"
    )
    vSopt: Optional[list] = Field(
        [35.0, 15.0, 0.0], description="optimal salinity for veg growth"
    )
    vInun: Optional[list] = Field(
        [1.0, 1.0, 1.0],
        description="reference value for inundation stress (nondimensional)",
    )
    ivNs: Optional[int] = Field(
        1, description="N limitation on veg growth(0: OFF; 1: ON)"
    )
    ivPs: Optional[int] = Field(
        1, description="P limitation on veg growth(0: OFF; 1: ON)"
    )
    ivMRT: Optional[int] = Field(0, description="veg mortality term (0: OFF;  1: ON)")
    vTMR: Optional[list] = Field(
        [17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
        description="reference temp(3,2) for leaf/stem mortality",
    )
    vKTMR: Optional[list] = Field(
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        description="temp dependence(3,2) for leaf/stem mortality",
    )
    vMR0: Optional[list] = Field(
        [12.8, 12.8, 12.8, 12.8, 12.8, 12.8],
        description="base value(3,2) of temp effect on mortality (unit: None)",
    )
    vMRcr: Optional[list] = Field(
        [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        description="reference value(3,2) for computing mortality (unit: None)",
    )
    valpha: Optional[list] = Field(
        [0.006, 0.006, 0.006],
        description="Initial slope of P-I curve (3 values for different vegetation types)",
    )
    vKe: Optional[list] = Field(
        [0.045, 0.045, 0.045], description="light attenuation from veg absorption"
    )
    vht0: Optional[list] = Field(
        [0.054, 0.054, 0.054],
        description="Base vegetation canopy height (3 values for different vegetation types)",
    )
    vcrit: Optional[list] = Field(
        [250.0, 250.0, 250.0],
        description="Critical mass for computing vegetation height (3 values for different vegetation types)",
    )
    v2ht: Optional[list] = Field(
        [0.0036, 0.0036, 0.0036, 0.001, 0.001, 0.001],
        description="Coefficients to convert mass to canopy height (6 values: 3 for each vegetation type, 2 for different conditions)",
    )
    vc2dw: Optional[list] = Field(
        [0.38, 0.38, 0.38],
        description="Carbon to dry weight ratio of vegetation (3 values for different vegetation types)",
    )
    v2den: Optional[list] = Field(
        [10, 10, 10],
        description="Coefficient for computing vegetation density (3 values for different vegetation types)",
    )
    vp2c: Optional[list] = Field(
        [0.01, 0.01, 0.01],
        description="Phosphorus to carbon ratio in vegetation (3 values for different vegetation types)",
    )
    vn2c: Optional[list] = Field(
        [0.09, 0.09, 0.09],
        description="Nitrogen to carbon ratio in vegetation (3 values for different vegetation types)",
    )
    vo2c: Optional[list] = Field(
        [2.67, 2.67, 2.67],
        description="Oxygen to carbon ratio in vegetation (3 values for different vegetation types)",
    )

    @field_validator("vpatch0")
    @classmethod
    def check_vpatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("vpatch0 must be either 1 or -999")
        return v

    @field_validator("vtleaf0")
    @classmethod
    def check_vtleaf0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vtleaf0 must be a list of 3 positive numbers")
        return v

    @field_validator("vtstem0")
    @classmethod
    def check_vtstem0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vtstem0 must be a list of 3 positive numbers")
        return v

    @field_validator("vtroot0")
    @classmethod
    def check_vtroot0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vtroot0 must be a list of 3 positive numbers")
        return v

    @field_validator("valpha")
    @classmethod
    def check_valpha(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("valpha must be a list of 3 positive numbers")
        return v

    @field_validator("vht0")
    @classmethod
    def check_vht0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vht0 must be a list of 3 positive numbers")
        return v

    @field_validator("vcrit")
    @classmethod
    def check_vcrit(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vcrit must be a list of 3 positive numbers")
        return v

    @field_validator("v2ht")
    @classmethod
    def check_v2ht(cls, v):
        if len(v) != 6 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("v2ht must be a list of 6 positive numbers")
        return v

    @field_validator("vc2dw")
    @classmethod
    def check_vc2dw(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vc2dw must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("v2den")
    @classmethod
    def check_v2den(cls, v):
        if len(v) != 3 or not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("v2den must be a list of 3 positive integers")
        return v

    @field_validator("vp2c")
    @classmethod
    def check_vp2c(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vp2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("vn2c")
    @classmethod
    def check_vn2c(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vn2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("vo2c")
    @classmethod
    def check_vo2c(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vo2c must be a list of 3 positive numbers")
        return v

    @model_validator(mode="after")
    def check_vfcp_sum(self):
        for i in range(3):
            if abs(sum(self.vfcp[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfcp values for vegetation type {i+1} must be 1"
                )
        return self

    @model_validator(mode="after")
    def check_vfnm_sum(self):
        for i in range(3):
            if abs(sum(self.vfnm[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfnm values for vegetation type {i+1} must be 1"
                )
        return self

    @model_validator(mode="after")
    def check_vfpm_sum(self):
        for i in range(3):
            if abs(sum(self.vfpm[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfpm values for vegetation type {i+1} must be 1"
                )
        return self

    @model_validator(mode="after")
    def check_vfcm_sum(self):
        for i in range(3):
            if abs(sum(self.vfcm[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfcm values for vegetation type {i+1} must be 1"
                )
        return self


class Bag(NamelistBaseModel):
    gpatch0: Optional[int] = Field(
        -999,
        description="Region flag for Benthic Algae (BA). 1 indicates BA is ON for all elements, -999 indicates spatial distribution.",
    )
    BA0: Optional[float] = Field(5.0, description="initial BA concentration (g[C].m-2)")
    gGPM: Optional[float] = Field(2.25, description="BA maximum growth rate (day-1)")
    gTGP: Optional[float] = Field(20.0, description="optimal temp. for BA growth (oC)")
    gKTGP: Optional[list] = Field(
        [0.004, 0.006], description="temp. dependence for BA growth (oC-2)"
    )
    gMTB: Optional[float] = Field(
        0.05, description="respiration rate at temp. of gTR (day-1)"
    )
    gPRR: Optional[float] = Field(
        0.1, description="predation rate at temp. of gTR (day-1)"
    )
    gTR: Optional[float] = Field(
        20.0, description="reference temperature for BA respiration (oC)"
    )
    gKTR: Optional[float] = Field(
        0.069, description="temp. dependence for BA respiration (oC-1)"
    )
    galpha: Optional[float] = Field(
        0.1,
        description="Initial slope of the Photosynthesis-Irradiance (P-I) curve for Benthic Algae (BA) in square meters per Einstein.",
    )
    gKSED: Optional[float] = Field(
        0.0, description="light attenuation due to sediment (None)"
    )
    gKBA: Optional[float] = Field(
        0.01, description="light attenuation coef. due to BA self-shading (g[C]-1.m2)"
    )
    gKhN: Optional[float] = Field(
        0.01, description="nitrogen half saturation for BA growth (g[N]/m2)"
    )
    gKhP: Optional[float] = Field(
        0.001, description="phosphorus half saturation for BA growth (g[P]/2)"
    )
    gp2c: Optional[float] = Field(
        0.0167, description="Phosphorus to carbon ratio in Benthic Algae (BA) biomass."
    )
    gn2c: Optional[float] = Field(
        0.167, description="Nitrogen to carbon ratio in Benthic Algae (BA) biomass."
    )
    go2c: Optional[float] = Field(
        2.67, description="Oxygen to carbon ratio for Benthic Algae (BA) respiration."
    )
    gFCP: Optional[list] = Field(
        [0.5, 0.45, 0.05],
        description="fraction of predation BA C into 3G classes in sediment",
    )
    gFNP: Optional[list] = Field(
        [0.5, 0.45, 0.05],
        description="fraction of predation BA N into 3G classes in sediment",
    )
    gFPP: Optional[list] = Field(
        [0.5, 0.45, 0.05],
        description="fraction of predation BA P into 3G classes in sediment",
    )

    @field_validator("gpatch0")
    @classmethod
    def check_gpatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("gpatch0 must be either 1 or -999")
        return v

    @field_validator("galpha")
    @classmethod
    def check_galpha(cls, v):
        if v <= 0:
            raise ValueError("galpha must be positive")
        return v

    @field_validator("gp2c")
    @classmethod
    def check_gp2c(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("gp2c must be between 0 and 1")
        return v

    @field_validator("gn2c")
    @classmethod
    def check_gn2c(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("gn2c must be between 0 and 1")
        return v

    @field_validator("go2c")
    @classmethod
    def check_go2c(cls, v):
        if v <= 0:
            raise ValueError("go2c must be positive")
        return v


class Ero(NamelistBaseModel):
    ierosion: Optional[int] = Field(
        0,
        description="Determines the type of benthic erosion flux to be modeled. 0: No erosion, 1: H2S flux, 2: POC (Particulate Organic Carbon) flux, 3: H2S flux (alternative method).",
    )

    @field_validator("ierosion")
    @classmethod
    def validate_ierosion(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("ierosion must be 0, 1, 2, or 3")
        return v


class Poc(NamelistBaseModel):
    erosion: Optional[int] = Field(
        864, description="Erosion rate in kilograms per square meter per day"
    )
    etau: Optional[str] = Field(
        "1.e-6", description="Critical bottom shear stress in Pascals"
    )
    eporo: Optional[float] = Field(
        0.8, description="Coefficient in erosion formula (see code in icm_sfm.F90)"
    )
    efrac: Optional[float] = Field(
        0.5,
        description="Fraction coefficient in erosion formula (see code in icm_sfm.F90)",
    )
    ediso: Optional[float] = Field(
        2.5, description="H2S erosion coefficient (see code in icm_sfm.F90)"
    )
    dfrac: Optional[list] = Field(
        [0.02, 0.02],
        description="Deposition fraction of POC (Particulate Organic Carbon). If negative, dfrac will be computed",
    )
    dWS_POC: Optional[list] = Field(
        [3.0, 3.0], description="coefficient in POC erosion (see code in icm_sfm.F90)"
    )

    @field_validator("erosion")
    @classmethod
    def validate_erosion(cls, v):
        if v <= 0:
            raise ValueError("Erosion rate must be positive")
        return v

    @field_validator("etau")
    @classmethod
    def validate_etau(cls, v):
        if v <= 0:
            raise ValueError("Critical bottom shear stress must be positive")
        return v

    @field_validator("eporo")
    @classmethod
    def validate_eporo(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Eporo must be between 0 and 1")
        return v

    @field_validator("efrac")
    @classmethod
    def validate_efrac(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Efrac must be between 0 and 1")
        return v

    @field_validator("ediso")
    @classmethod
    def validate_ediso(cls, v):
        if v <= 0:
            raise ValueError("Ediso must be positive")
        return v

    @field_validator("dfrac")
    @classmethod
    def validate_dfrac(cls, v):
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError("Dfrac must be a list of two float values")
        return v


class Icm(NamelistBaseModel):
    marco: Optional[Marco] = Field(default_factory=Marco)
    core: Optional[Core] = Field(default_factory=Core)
    sfm: Optional[Sfm] = Field(default_factory=Sfm)
    silica: Optional[Silica] = Field(default_factory=Silica)
    zb: Optional[Zb] = Field(default_factory=Zb)
    ph_icm: Optional[Ph_icm] = Field(default_factory=Ph_icm)
    sav: Optional[Sav] = Field(default_factory=Sav)
    stem: Optional[Stem] = Field(default_factory=Stem)
    veg: Optional[Veg] = Field(default_factory=Veg)
    bag: Optional[Bag] = Field(default_factory=Bag)
    ero: Optional[Ero] = Field(default_factory=Ero)
    poc: Optional[Poc] = Field(default_factory=Poc)
