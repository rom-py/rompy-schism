# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import Optional

from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Marco(NamelistBaseModel):
    nsub: Optional[int] = Field(None, description="")
    ike: Optional[int] = Field(None, description="")
    ke0: Optional[float] = Field(
        None, description="backgroud light extinction coefficient (1/m)"
    )
    kec: Optional[float] = Field(None, description="Light attenu. due to chlorophyll")
    kes: Optional[float] = Field(None, description="Light attenu. due to TSS")
    kesalt: Optional[float] = Field(
        None, description="Light attenu. due to CDOM (related to salinity)"
    )
    tss2c: Optional[float] = Field(None, description="TSS to carbon ratio")
    ilight: Optional[int] = Field(None, description="")
    alpha: Optional[list] = Field(
        None, description="init. slope of P-I curve (g[C]*m2/g[Chl]/E), (iLight=0)"
    )
    ipr: Optional[int] = Field(
        None, description="(0: linear formulation; 1: quadratic)"
    )
    prr: Optional[list] = Field(
        None,
        description="predation rate by higher trophic level (day-1, or day-1.g-1.m3)",
    )
    wqc0: Optional[list] = Field(None, description="")
    wsp: Optional[list] = Field(None, description="")
    wspn: Optional[list] = Field(None, description="")
    isilica: Optional[int] = Field(None, description="")
    izb: Optional[int] = Field(None, description="")
    iph: Optional[int] = Field(None, description="")
    icbp: Optional[int] = Field(None, description="")
    isav_icm: Optional[int] = Field(None, description="")
    iveg_icm: Optional[int] = Field(None, description="0: VEG off;  1: VEG on")
    ised: Optional[int] = Field(None, description="")
    iba: Optional[int] = Field(None, description="")
    irad: Optional[int] = Field(None, description="")
    isflux: Optional[int] = Field(None, description="")
    ibflux: Optional[int] = Field(None, description="")
    iout_icm: Optional[int] = Field(None, description="")
    nspool_icm: Optional[int] = Field(None, description="")
    ilimit: Optional[int] = Field(None, description="")
    idry_icm: Optional[int] = Field(None, description="")


class Core(NamelistBaseModel):
    gpm: Optional[list] = Field(None, description="PB growth rates (day-1)")
    tgp: Optional[list] = Field(None, description="optimal temp. for PB growth (oC)")
    ktgp: Optional[list] = Field(
        None, description="temp. dependence for PB growth; dim(PB=1:3,1:2) (oC-2)"
    )
    mtr: Optional[list] = Field(
        None, description="PB photorespiration coefficient (0<MTR<1)"
    )
    mtb: Optional[list] = Field(None, description="PB metabolism rates (day-1)")
    tmt: Optional[list] = Field(
        None, description="reference temp. for PB metabolism (oC)"
    )
    ktmt: Optional[list] = Field(
        None, description="temp. dependence for PB metabolism (oC-1)"
    )
    fcp: Optional[list] = Field(
        None,
        description="fractions of PB carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,3)",
    )
    fnp: Optional[list] = Field(
        None, description="fractions of PB nitrogen into (RPON,LPON,DON,NH4,SRPON)"
    )
    fpp: Optional[list] = Field(
        None, description="fractions of PB Phosphorus into (RPOP,LPOP,DOP,PO4,SRPOP)"
    )
    fcm: Optional[list] = Field(
        None,
        description="fractions of PB metabolism carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,4)",
    )
    fnm: Optional[list] = Field(
        None,
        description="fractions of PB metabolism N. into (RPON,LPON,DON,NH4,SRPON); dim(PB=1:3,5)",
    )
    fpm: Optional[list] = Field(
        None,
        description="fractions of PB metabolism P. into (RPOP,LPOP,DOP,PO4,SRPOP); dim(PB=1:3,5)",
    )
    nit: Optional[float] = Field(None, description="maximum nitrification rate (day-1)")
    tnit: Optional[float] = Field(
        None, description="optimal temp. for nitrification (oC)"
    )
    ktnit: Optional[list] = Field(
        None, description="temp. dependence (T<=TNit & T>TNit) for nitrification (oC-2)"
    )
    khdon: Optional[float] = Field(
        None, description="DO half saturation for nitrification (mg/L)"
    )
    khdoox: Optional[float] = Field(
        None,
        description="DO half saturation for dentrification & DOC's oxic respiration (mg/L)",
    )
    khno3dn: Optional[float] = Field(
        None, description="NO3 half saturation for denitrification (mg/L)"
    )
    kc0: Optional[list] = Field(
        None, description="minimum decay rate of RPOC,LPOC,DOC (day-1)"
    )
    kn0: Optional[list] = Field(
        None, description="minimum decay rate of RPON,LPON,DON (day-1)"
    )
    kp0: Optional[list] = Field(
        None, description="minimum decay rate of RPOP,LPOP,DOP (day-1)"
    )
    kcalg: Optional[list] = Field(
        None, description="algae effect on RPOC,LPOC,DOC decay (day-1.m3.g[C]-1)"
    )
    knalg: Optional[list] = Field(
        None, description="algae effect on RPON,LPON,DON decay (day-1.m3.g[C]-1)"
    )
    kpalg: Optional[list] = Field(
        None, description="algae effect on RPOP,LPOP,DOP decay (day-1.m3.g[C]-1)"
    )
    trm: Optional[list] = Field(
        None, description="reference temp. for (RPOM,LPOM,DOM) decay (oC)"
    )
    ktrm: Optional[list] = Field(
        None, description="temp. dependence for (RPOM,LPOM,DOM) decay (oC-1)"
    )
    ksr0: Optional[list] = Field(
        None, description="decay rates of SRPOC,SRPON,SRPOP (day-1)"
    )
    trsr: Optional[list] = Field(
        None, description="reference temp. for (SRPOC,SRPON,SRPOP) decay (oC)"
    )
    ktrsr: Optional[list] = Field(
        None, description="temp. dependence for (SRPOC,SRPON,SRPOP) decay (oC-1)"
    )
    kpip: Optional[float] = Field(None, description="dissolution rate of PIP (day-1)")
    kcd: Optional[float] = Field(
        None, description="oxidation rate of COD at TRCOD (day-1)"
    )
    trcod: Optional[float] = Field(
        None, description="reference temp. for COD oxidation (oC)"
    )
    ktrcod: Optional[float] = Field(
        None, description="temp. dependence for COD oxidation (oC-1)"
    )
    khcod: Optional[float] = Field(
        None, description="COD half saturation for COD oxidation (mg[O2]/L)"
    )
    khn: Optional[list] = Field(None, description="nitrogen half saturation (mg/L)")
    khp: Optional[list] = Field(None, description="phosphorus half saturation (mg/L)")
    khsal: Optional[list] = Field(
        None,
        description="salinity when PB growth is halved (PSU); (1e6: no salinity stress)",
    )
    c2chl: Optional[list] = Field(
        None, description="carbon to chlorophyll ratio (g[C]/mg[Chl])"
    )
    n2c: Optional[list] = Field(
        None, description="nitrogen to carbon ratio for phytoplankton"
    )
    p2c: Optional[list] = Field(
        None, description="phosphorus to carbon ratio for phytoplankton"
    )
    o2c: Optional[float] = Field(
        None, description="oxygen to carbon ratio in respiration"
    )
    o2n: Optional[float] = Field(
        None, description="oxygen to ammonium ratio (g[O2]/g[NH4])"
    )
    dn2c: Optional[float] = Field(
        None,
        description="mass of NO3 consumed per mass of DOC oxidized in denit. (g[N]/g[C])",
    )
    an2c: Optional[float] = Field(
        None, description="ratio of denit. rate to oxic DOC respiration rate"
    )
    khdo: Optional[list] = Field(
        None, description="DO half saturation for PB's DOC excretion (mg/L)"
    )
    kpo4p: Optional[float] = Field(
        None, description="coefficient relating PO4 sorption to TSS"
    )
    wrea: Optional[float] = Field(
        None, description="baseline wind-induced reaeration coefficient for DO (day-1)"
    )
    pbmin: Optional[list] = Field(
        None, description="minimum PB concentration (mg[C]/L)"
    )
    dz_flux: Optional[list] = Field(
        None,
        description="surface/bottom thikness (m) within which sflux/bflux are redistributed",
    )


class Sfm(NamelistBaseModel):
    btemp0: Optional[float] = Field(None, description="initial temp. (oC)")
    bstc0: Optional[float] = Field(None, description="surface transfer coefficient")
    bstr0: Optional[float] = Field(None, description="benthic stress (day)")
    bthp0: Optional[float] = Field(None, description="consective days of hypoxia (day)")
    btox0: Optional[float] = Field(
        None, description="consective days of oxic condition after hypoxia event (day)"
    )
    bnh40: Optional[float] = Field(None, description="NH4")
    bno30: Optional[float] = Field(None, description="NO3")
    bpo40: Optional[float] = Field(None, description="PO4")
    bh2s0: Optional[float] = Field(None, description="H2S")
    bch40: Optional[float] = Field(None, description="CH4")
    bpos0: Optional[float] = Field(None, description="POS")
    bsa0: Optional[float] = Field(None, description="SA")
    bpoc0: Optional[list] = Field(None, description="POC(G=1:3)")
    bpon0: Optional[list] = Field(None, description="PON(G=1:3)")
    bpop0: Optional[list] = Field(None, description="POP(G=1:3)")
    bdz: Optional[float] = Field(None, description="sediment thickness (m)")
    bvb: Optional[str] = Field(
        None, description="burial rate (m.day-1); 1 cm/yr=2.74e-5 m.day-1"
    )
    bsolid: Optional[list] = Field(
        None, description="sediment solid conc. in Layer 1 and Layer 2 (Kg.L-1)"
    )
    bdiff: Optional[str] = Field(
        None, description="diffusion coefficient for sediment temp. (m2/s)"
    )
    btr: Optional[int] = Field(
        None, description="reference temp. for sediment processes"
    )
    bvpmin: Optional[str] = Field(
        None, description="minimum particle mixing velocity coefficient (m.day-1)"
    )
    bvp: Optional[str] = Field(
        None, description="particle mixing velocity coefficient (m.day-1)"
    )
    bvd: Optional[str] = Field(
        None, description="diffusion velocity coefficient (m.day-1)"
    )
    bktvp: Optional[float] = Field(
        None, description="temp. dependece of particle mixing velocity"
    )
    bktvd: Optional[float] = Field(
        None, description="temp. dependece of diffusion velocity"
    )
    bkst: Optional[float] = Field(
        None, description="1st order decay rate of benthic stress  (day-1)"
    )
    bstmax: Optional[float] = Field(
        None,
        description="maximum value of benthic stress (day) (note: smaller than 1/bKST)",
    )
    bkhdo_vp: Optional[float] = Field(
        None, description="DO half-saturation of particle mixing (mg/L)"
    )
    bdoc_st: Optional[float] = Field(
        None, description="DO criteria for benthic stress (mg/L)"
    )
    banoxic: Optional[float] = Field(
        None,
        description="consective days of hypoxia causing maximum benthic stress (day)",
    )
    boxic: Optional[float] = Field(
        None, description="time lag for bethos recovery from hypoxia event (day)"
    )
    bp2d: Optional[float] = Field(
        None,
        description="ratio from mixing coef. to diffusion coef. (benthos enhanced effect)",
    )
    bkc: Optional[list] = Field(
        None, description="decay rate of POC (3G class) at bTR (day-1)"
    )
    bkn: Optional[list] = Field(
        None, description="decay rate of PON (3G class) at bTR (day-1)"
    )
    bkp: Optional[list] = Field(
        None, description="decay rate of POP (3G class) at bTR (day-1)"
    )
    bktc: Optional[list] = Field(
        None, description="temp. dependence of POC decay (oC-1)"
    )
    bktn: Optional[list] = Field(
        None, description="temp. dependence of PON decay (oC-1)"
    )
    bktp: Optional[list] = Field(
        None, description="temp. dependence of POP decay (oC-1)"
    )
    bfcp: Optional[list] = Field(None, description="Phyto POC into sed POC (G3,PB=1:3)")
    bfnp: Optional[list] = Field(None, description="Phyto PON into sed PON (G3,PB=1:3)")
    bfpp: Optional[list] = Field(None, description="Phyto POP into sed POP (G3,PB=1:3)")
    bfcm: Optional[list] = Field(None, description="refractory POC into sed POC(3G)")
    bfnm: Optional[list] = Field(None, description="refractory PON into sed PON(3G)")
    bfpm: Optional[list] = Field(None, description="refractory POP into sed POP(3G)")
    bknh4f: Optional[float] = Field(
        None, description="NH4 reaction rate in freshwater  at bTR (1st layer) (m/day)"
    )
    bknh4s: Optional[float] = Field(
        None, description="NH4 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bktnh4: Optional[float] = Field(
        None, description="temp. dependency for NH4 reaction (oC-1)"
    )
    bkhnh4: Optional[float] = Field(
        None, description="half-stauration NH4 for nitrification (g/m3)"
    )
    bkhdo_nh4: Optional[float] = Field(
        None, description="half-stauration DO for nitrification (g/m3)"
    )
    bpienh4: Optional[float] = Field(
        None, description="partition coefficients of NH4 in Layer 1 & 2 (Kg-1.L)"
    )
    bsaltn: Optional[float] = Field(
        None,
        description="salinity criteria of fresh/salty water for NH4/NO3 reaction (PSU)",
    )
    bkno3f: Optional[float] = Field(
        None, description="NO3 reaction rate in freshwater at bTR (1st layer) (m/day)"
    )
    bkno3s: Optional[float] = Field(
        None, description="NO3 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bkno3: Optional[float] = Field(
        None, description="NO3 reaction rate (2nd layer) (m/day)"
    )
    bktno3: Optional[float] = Field(
        None, description="temp. dependency for NO3 reaction (oC-1)"
    )
    bkh2sd: Optional[float] = Field(
        None, description="dissolved H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bkh2sp: Optional[float] = Field(
        None, description="particulate H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bkth2s: Optional[float] = Field(
        None, description="temp. dependency for H2S reaction (oC-1)"
    )
    bpieh2ss: Optional[float] = Field(
        None, description="partition coefficient of NH4 in Layer 1 (Kg-1.L)"
    )
    bpieh2sb: Optional[float] = Field(
        None, description="partition coefficient of NH4 in Layer 2 (Kg-1.L)"
    )
    bkhdo_h2s: Optional[float] = Field(
        None, description="O2 constant to normalize H2S oxidation (g[O2]/m3)"
    )
    bsaltc: Optional[float] = Field(
        None,
        description="salinity criteria of fresh/salty water for carbon reaction (PSU)",
    )
    bkch4: Optional[float] = Field(
        None, description="CH4 reaction rate at bTR (1st layer) (m/day)"
    )
    bktch4: Optional[float] = Field(
        None, description="temp. dependency for CH4 reaction"
    )
    bkhdo_ch4: Optional[float] = Field(
        None, description="half-saturation DO for CH4 oxidation (g[O2]/m3)"
    )
    bo2n: Optional[float] = Field(
        None, description="oxygen to nitrogen ratio in sediment (denitrification)"
    )
    bpiepo4: Optional[float] = Field(
        None, description="partition coefficient of PO4 in Layer 2 (Kg-1.L)"
    )
    bkopo4f: Optional[float] = Field(
        None, description="oxygen dependency for PO4 sorption in freshwater in Layer 1"
    )
    bkopo4s: Optional[float] = Field(
        None, description="oxygen dependency for PO4 sorption in salty water in Layer 1"
    )
    bdoc_po4: Optional[float] = Field(
        None, description="DO criteria for PO4 sorptiona (g[O2]/m3)"
    )
    bsaltp: Optional[float] = Field(
        None,
        description="salinity criteria of fresh/salty water for PO4 partition (PSU)",
    )
    bks: Optional[float] = Field(
        None, description="decay rate of POS (3G class) at bTR"
    )
    bkts: Optional[float] = Field(None, description="temp. dependence of POS decay")
    bsisat: Optional[float] = Field(
        None, description="silica saturation conc. in pore water (g[Si]/m3)"
    )
    bpiesi: Optional[float] = Field(
        None, description="partition coefficient of silica in Layer 2 (Kg-1.L)"
    )
    bkosi: Optional[float] = Field(
        None, description="oxygen dependency for silica sorption in Layer 1"
    )
    bkhpos: Optional[str] = Field(
        None, description="POS half saturation for POS dissolution (g/m3)"
    )
    bdoc_si: Optional[float] = Field(
        None, description="DO criteria for silica sorptiona (g[O2]/m3)"
    )
    bjposa: Optional[float] = Field(
        None,
        description="additional POS flux associated with POM detrius beside algea (g.m-2.day-1)",
    )
    bfcs: Optional[list] = Field(None, description="SAV POC into 3G sed 3G POC")
    bfns: Optional[list] = Field(None, description="SAV PON into 3G sed 3G PON")
    bfps: Optional[list] = Field(None, description="SAV POP into 3G sed 3G POP")
    bfcv: Optional[list] = Field(None, description="VEG POC into sed POC (G3,PB=1:3)")
    bfnv: Optional[list] = Field(None, description="VEG PON into sed PON (G3,PB=1:3)")
    bfpv: Optional[list] = Field(None, description="VEG POP into sed POP (G3,PB=1:3)")


class Silica(NamelistBaseModel):
    fsp: Optional[list] = Field(
        None, description="fractions of diatom silica into (SU,SA)"
    )
    fsm: Optional[list] = Field(
        None, description="fractions of diatom metabolism Si into (SU,SA)"
    )
    ks: Optional[float] = Field(
        None, description="dissolution rate of SU at TRS (day-1)"
    )
    trs: Optional[float] = Field(
        None, description="reference temp. for SU dissolution (oC)"
    )
    ktrs: Optional[float] = Field(
        None, description="temp. dependence for SU dissolution (oC-1)"
    )
    khs: Optional[list] = Field(
        None, description="silica half saturation (mg/L); (0.0: no Si limitation)"
    )
    s2c: Optional[list] = Field(
        None, description="silica to carbon ratio for phytolankton; (0.0: no Si uptake)"
    )
    ksap: Optional[float] = Field(
        None, description="coefficient relating Silicate(SA) sorption to TSS"
    )


class Zb(NamelistBaseModel):
    zgpm: Optional[list] = Field(
        None, description="ZB predation rate(day-1); dim(prey=1:8, ZB=1:2)"
    )
    zkhg: Optional[list] = Field(
        None, description="reference prey conc.(mg/L); dim(prey=1:8, ZB=1:2)"
    )
    ztgp: Optional[list] = Field(None, description="optimal temp. for ZB growth (oC)")
    zktgp: Optional[list] = Field(
        None,
        description="temp. dependence for ZB growth (T<=zTGP & T>zTGP); dim(ZB=1:2,1:2) (oC-2)",
    )
    zag: Optional[float] = Field(
        None, description="ZB assimilation efficiency ratio (0-1)"
    )
    zrg: Optional[float] = Field(
        None, description="ZB respiration ratio when it grazes (0-1)"
    )
    zmrt: Optional[list] = Field(None, description="ZB mortality rates (day-1)")
    zmtb: Optional[list] = Field(None, description="ZB metabolism rates (day-1)")
    ztmt: Optional[list] = Field(
        None, description="reference temp. for ZB metabolism (oC)"
    )
    zktmt: Optional[list] = Field(
        None, description="temp. dependence for ZB metabolism (oC-1)"
    )
    zfcp: Optional[list] = Field(
        None, description="fractions of ZB carbon into (RPOC,LPOC,DOC)"
    )
    zfnp: Optional[list] = Field(
        None, description="fractions of ZB nitrogen into (RPON,LPON,DON,NH4)"
    )
    zfpp: Optional[list] = Field(
        None, description="fractions of ZB Phosphorus into (RPOP,LPOP,DOP,PO4)"
    )
    zfsp: Optional[list] = Field(
        None, description="fractions of ZB silica into (SU,SA)"
    )
    zfcm: Optional[list] = Field(
        None, description="fractions of ZB metabolism carbon into DOC; dim(ZB=1:2)"
    )
    zfnm: Optional[list] = Field(
        None,
        description="fractions of ZB metabolism N. into (RPON,LPON,DON,NH4); dim(ZB=1:2,4)",
    )
    zfpm: Optional[list] = Field(
        None,
        description="fractions of ZB metabolism P. into (RPOP,LPOP,DOP,PO4); dim(ZB=1:2,4)",
    )
    zfsm: Optional[list] = Field(
        None, description="fractions of ZB metabolism Si into (SU,SA); dim(ZB=1:2,2)"
    )
    zkhdo: Optional[list] = Field(
        None, description="DO half saturation for ZB's DOC excretion (mg/L)"
    )
    zn2c: Optional[list] = Field(
        None, description="nitrogen to carbon ratio for zooplankton"
    )
    zp2c: Optional[list] = Field(
        None, description="phosphorus to carbon ratio for zooplankton"
    )
    zs2c: Optional[list] = Field(
        None, description="silica to carbon ratio for zooplankton"
    )
    z2pr: Optional[list] = Field(
        None,
        description="ratio converting ZB+PB biomass to predation rates on ZB (L.mg-1.day-1)",
    )
    p2pr: Optional[float] = Field(
        None,
        description="ratio converting ZB+PB biomass to predation rates on PB (L.mg-1.day-1)",
    )


class Ph_icm(NamelistBaseModel):
    ppatch0: Optional[int] = Field(
        None, description="region flag for pH (1: ON all elem.; -999: spatial)"
    )
    pkcaco3: Optional[float] = Field(
        None, description="dissulution bewteen CaCO3 and Ca++"
    )
    pkca: Optional[float] = Field(
        None, description="sediment surface transfer coefficient from CaCO3 to Ca++"
    )
    prea: Optional[float] = Field(None, description="reaeration rate for CO2")
    inu_ph: Optional[int] = Field(None, description="nudge option for pH model")


class Sav(NamelistBaseModel):
    spatch0: Optional[int] = Field(
        None, description="region flag for SAV. (1: ON all elem.; -999: spatial)"
    )
    stleaf0: Optional[int] = Field(None, description="init conc of total sav leaf")
    ststem0: Optional[int] = Field(None, description="init conc of total sav stem")
    stroot0: Optional[int] = Field(None, description="init conc of total sav root")
    sgpm: Optional[float] = Field(None, description="maximum growth rate (day-1)")
    stgp: Optional[int] = Field(None, description="optimal growth temperature (oC)")
    sktgp: Optional[list] = Field(
        None, description="temp. dependence for growth (T<=sTGP & T>sTGP)"
    )
    sfam: Optional[float] = Field(
        None, description="fraction of leaf production to active metabolism"
    )
    sfcp: Optional[list] = Field(
        None, description="fractions of production to leaf/stem/root biomass"
    )
    smtb: Optional[list] = Field(None, description="metabolism rates of leaf/stem/root")
    stmt: Optional[list] = Field(
        None, description="reference temp. for leaf/stem/root metabolism"
    )
    sktmt: Optional[list] = Field(
        None, description="temp. dependence of leaf/stem/root metabolism"
    )
    sfcm: Optional[list] = Field(
        None, description="fractions of metabolism C into (RPOC,LPOC,DOC,CO2)"
    )
    sfnm: Optional[list] = Field(
        None, description="fractions of metabolism N into (RPON,LPON,DON,NH4)"
    )
    sfpm: Optional[list] = Field(
        None, description="fractions of metabolism P into (RPOP,LPOP,DOP,PO4)"
    )
    skhnw: Optional[float] = Field(
        None, description="nitrogen half saturation in water column"
    )
    skhns: Optional[float] = Field(
        None, description="nitrogen half saturation in sediments"
    )
    skhnh4: Optional[float] = Field(None, description="ammonium half saturation")
    skhpw: Optional[float] = Field(
        None, description="phosphorus half saturation in water column"
    )
    skhps: Optional[float] = Field(
        None, description="phosphorus half saturation in sediments"
    )
    salpha: Optional[float] = Field(
        None, description="init. slope of P-I curve (g[C]*m2/g[Chl]/E)"
    )
    ske: Optional[float] = Field(
        None, description="light attenuation due to sav absorption"
    )
    shtm: Optional[list] = Field(
        None, description="minimum (base) and  maximium canopy height"
    )
    s2ht: Optional[list] = Field(
        None, description="coeffs. converting (leaf,stem,root) to canopy height"
    )
    sc2dw: Optional[float] = Field(
        None, description="carbon to dry weight ratio of SAV"
    )
    s2den: Optional[int] = Field(
        None, description="coeff. computing SAV density from leaf"
    )


class Stem(NamelistBaseModel):
    sn2c: Optional[float] = Field(None, description="nitrogen to carbon ratio of sav")
    sp2c: Optional[float] = Field(None, description="phosphorus to carbon ratio")
    so2c: Optional[float] = Field(None, description="oxygen to carbon ratio")


class Veg(NamelistBaseModel):
    vpatch0: Optional[int] = Field(
        None, description="region flag for VEG. (1: ON all elem.; -999: spatial)"
    )
    vtleaf0: Optional[list] = Field(None, description="init conc to total veg leaf")
    vtstem0: Optional[list] = Field(None, description="init conc to total veg stem")
    vtroot0: Optional[list] = Field(None, description="init conc to total veg root")
    vgpm: Optional[list] = Field(None, description="maximum growth rate (day-1)")
    vfam: Optional[list] = Field(
        None, description="fractions of leaf production to active metabolism"
    )
    vtgp: Optional[list] = Field(None, description="optimal growth temperature (oC)")
    vktgp: Optional[list] = Field(
        None, description="temp. dependence(3,2) for growth (T<=vTGP & T>vTGP)"
    )
    vfcp: Optional[list] = Field(
        None,
        description="fractions of production to leaf/stem/root biomass; dim(veg,leaf/stem/root)",
    )
    vmtb: Optional[list] = Field(
        None, description="metabolism rates of leaf/stem/root; dim(veg,leaf/stem/root)"
    )
    vtmt: Optional[list] = Field(
        None, description="reference temp. for leaf/stem/root metabolism"
    )
    vktmt: Optional[list] = Field(
        None, description="temp. depdenpence for leaf/stem/root metabolism"
    )
    vfnm: Optional[list] = Field(
        None, description="fractions(3,4) of metabolism N into (RPON,LPON,DON,NH4)"
    )
    vfpm: Optional[list] = Field(
        None, description="fractions(3,4) of metabolism P into (RPOP,LPOP,DOP,PO4)"
    )
    vfcm: Optional[list] = Field(
        None, description="fractions(3,4) of metabolism N into (RPOC,LPOC,DOC,CO2)"
    )
    ivnc: Optional[int] = Field(
        None, description="recycled veg N goes to (0: sediment; 1: water)"
    )
    ivpc: Optional[int] = Field(
        None, description="recycled veg P goes to (0: sediment; 1: water)"
    )
    vkhns: Optional[list] = Field(
        None, description="nitrogen half saturation in sediments"
    )
    vkhps: Optional[list] = Field(
        None, description="phosphorus half saturation in sediments"
    )
    vscr: Optional[list] = Field(
        None, description="reference sality for computing veg growth"
    )
    vsopt: Optional[list] = Field(None, description="optimal salinity for veg growth")
    vinun: Optional[list] = Field(
        None, description="reference value for inundation stress (nondimensional)"
    )
    ivns: Optional[int] = Field(
        None, description="N limitation on veg growth(0: OFF; 1: ON)"
    )
    ivps: Optional[int] = Field(
        None, description="P limitation on veg growth(0: OFF; 1: ON)"
    )
    ivmrt: Optional[int] = Field(
        None, description="veg mortality term (0: OFF;  1: ON)"
    )
    vtmr: Optional[list] = Field(
        None, description="reference temp(3,2) for leaf/stem mortality"
    )
    vktmr: Optional[list] = Field(
        None, description="temp dependence(3,2) for leaf/stem mortality"
    )
    vmr0: Optional[list] = Field(
        None, description="base value(3,2) of temp effect on mortality (unit: None)"
    )
    vmrcr: Optional[list] = Field(
        None, description="reference value(3,2) for computing mortality (unit: None)"
    )
    valpha: Optional[list] = Field(None, description="init. slope of P-I curve")
    vke: Optional[list] = Field(
        None, description="light attenuation from veg absorption"
    )
    vht0: Optional[list] = Field(None, description="base veg canopy hgt (vht)")
    vcrit: Optional[list] = Field(None, description="critical mass for computing vht")
    v2ht: Optional[list] = Field(
        None, description="coefs. convert mass(3,2) to canopy height"
    )
    vc2dw: Optional[list] = Field(None, description="carbon to dry weight ratio of VEG")
    v2den: Optional[list] = Field(None, description="coeff. computing veg density")
    vp2c: Optional[list] = Field(None, description="phosphorus to carbon ratio")
    vn2c: Optional[list] = Field(None, description="nitrogen to carbon ratio")
    vo2c: Optional[list] = Field(None, description="oxygen to carbon ratio")


class Bag(NamelistBaseModel):
    gpatch0: Optional[int] = Field(
        None, description="region flag for BA. (1: ON all elem.; -999: spatial)"
    )
    ba0: Optional[float] = Field(
        None, description="initial BA concentration (g[C].m-2)"
    )
    ggpm: Optional[float] = Field(None, description="BA maximum growth rate (day-1)")
    gtgp: Optional[float] = Field(None, description="optimal temp. for BA growth (oC)")
    gktgp: Optional[list] = Field(
        None, description="temp. dependence for BA growth (oC-2)"
    )
    gmtb: Optional[float] = Field(
        None, description="respiration rate at temp. of gTR (day-1)"
    )
    gprr: Optional[float] = Field(
        None, description="predation rate at temp. of gTR (day-1)"
    )
    gtr: Optional[float] = Field(
        None, description="reference temperature for BA respiration (oC)"
    )
    gktr: Optional[float] = Field(
        None, description="temp. dependence for BA respiration (oC-1)"
    )
    galpha: Optional[float] = Field(None, description="init. slope of P-I curve (m2/E)")
    gksed: Optional[float] = Field(
        None, description="light attenuation due to sediment (None)"
    )
    gkba: Optional[float] = Field(
        None, description="light attenuation coef. due to BA self-shading (g[C]-1.m2)"
    )
    gkhn: Optional[float] = Field(
        None, description="nitrogen half saturation for BA growth (g[N]/m2)"
    )
    gkhp: Optional[float] = Field(
        None, description="phosphorus half saturation for BA growth (g[P]/2)"
    )
    gp2c: Optional[float] = Field(None, description="phosphorus to carbon ratio")
    gn2c: Optional[float] = Field(None, description="nitrogen to carbon ratio")
    go2c: Optional[float] = Field(None, description="oxygen to carbon ratio")
    gfcp: Optional[list] = Field(
        None, description="fraction of predation BA C into 3G classes in sediment"
    )
    gfnp: Optional[list] = Field(
        None, description="fraction of predation BA N into 3G classes in sediment"
    )
    gfpp: Optional[list] = Field(
        None, description="fraction of predation BA P into 3G classes in sediment"
    )


class Ero(NamelistBaseModel):
    ierosion: Optional[int] = Field(
        None, description="1: H2S flux; 2: POC flux; 3:  H2S"
    )


class Poc(NamelistBaseModel):
    erosion: Optional[int] = Field(None, description="erosion rate (kg.m-2.day)")
    etau: Optional[str] = Field(None, description="critical bottom shear stress (Pa)")
    eporo: Optional[float] = Field(
        None, description="coefficinet in erosion formula (see code in icm_sfm.F90)"
    )
    efrac: Optional[float] = Field(
        None,
        description="fraction coefficinet in erosion formula (see code in icm_sfm.F90)",
    )
    ediso: Optional[float] = Field(
        None, description="H2S erosion coeffcient (see code in icm_sfm.F90)"
    )
    dfrac: Optional[list] = Field(
        None,
        description="deposition fraction of POC (negative value: dfrac will be computed)",
    )
    dws_poc: Optional[list] = Field(
        None, description="coefficient in POC erosion (see code in icm_sfm.F90)"
    )


class Icm(NamelistBaseModel):
    """

    The full contents of the namelist file are shown below providing
    associated documentation for the objects:

    !-----------------------------------------------------------------------
    ! ICM model parameter inputs.
    ! Format rules
    ! (1) Lines beginning with "!" are comments; blank lines are ignored;
    ! (2) one line for each parameter in the format: keywords= (value1,value2,...);
    !     keywords are case sensitive; spaces allowed between keywords and "=" and value;
    !     comments starting with "!"  allowed after value;
    ! (3) value is an integer, double, or string (no single quote needed); for double,
    !     any of the format is acceptable: 40 40. 4.e1; ! Use of decimal point in integers
    !     is OK but discouraged.
    ! (4) spatially varying parameters are available for most parameters when value=-999;
    !     spatial value will be read from "ICM_param.nc"
    !     dimension=(npt),(npt,d1),or (npt,d1,d2) for scalar/1D/2D param, where npt=ne/np
    !---------------------------------------------------------------------------------
    !---------------------------state variables in ICM--------------------------------
    !---------------------------------------------------------------------------------
    !Core Module
    !     1  PB1   :  Diatom                                     g/m^3
    !     2  PB2   :  Green Algae                                g/m^3
    !     3  PB3   :  Cyanobacteria                              g/m^3
    !     4  RPOC  :  Refractory Particulate Organic Carbon      g/m^3
    !     5  LPOC  :  Labile Particulate Organic Carbon          g/m^3
    !     6  DOC   :  Dissolved Orgnaic Carbon                   g/m^3
    !     7  RPON  :  Refractory Particulate Organic Nitrogen    g/m^3
    !     8  LPON  :  Labile Particulate Organic Nitrogen        g/m^3
    !     9  DON   :  Dissolved Orgnaic Nitrogen                 g/m^3
    !     10 NH4   :  Ammonium Nitrogen                          g/m^3
    !     11 NO3   :  Nitrate Nitrogen                           g/m^3
    !     12 RPOP  :  Refractory Particulate Organic Phosphorus  g/m^3
    !     13 LPOP  :  Labile Particulate Organic Phosphorus      g/m^3
    !     14 DOP   :  Dissolved Orgnaic Phosphorus               g/m^3
    !     15 PO4   :  Total Phosphate                            g/m^3
    !     16 COD   :  Chemical Oxygen Demand                     g/m^3
    !     17 DOX   :  Dissolved Oxygen                           g/m^3
    !Silica Module
    !     1  SU    :  Particulate Biogenic Silica                g/m^3
    !     2  SA    :  Available Silica                           g/m^3
    !Zooplankton Module
    !     1  ZB1   :  1st zooplankton                            g/m^3
    !     2  ZB2   :  2nd zooplankton                            g/m^3
    !pH Module
    !     1  TIC   :  Total Inorganic Carbon                     g/m^3
    !     2  ALK   :  Alkalinity                                 g[CaCO3]/m^3
    !     3  CA    :  Dissolved Calcium                          g[CaCO3]/m^3
    !     4  CACO3 :  Calcium Carbonate                          g[CaCO3]/m^3
    !CBP Module
    !     1  SRPOC :  Slow Refractory Particulate Organic Carbon g/m^3
    !     2  SRPON :  Slow Refractory Particulate Organic Nitro. g/m^3
    !     3  SRPOP :  Slow Refractory Particulate Organic Phosp. g/m^3
    !     4  PIP   :  Particulate Inorganic Phosphate            g/m^3
    !SAV Module (no transport variables)
    !VEG Module (no transport variables)
    !SFM Module (no transport variables)
    !BA  Module (no transport variables)
    !---------------------------------------------------------------------------------

    &MARCO
    !-----------------------------------------------------------------------
    !number of subcycles in ICM kinetics
    !-----------------------------------------------------------------------
    nsub=1

    !-----------------------------------------------------------------------
    !options of formulations for computing light attenuation coefficients
    !  iKe=0: Ke=Ke0+KeC*Chl+KeS*(tss2c*POC);  TSS=tss2c*POC
    !  iKe=1: Ke=Ke0+KeC*Chl+KeS*TSS;        from SED3D or saved total_sed_conc
    !  iKe=2: Ke=Ke0+KeC*Chl+KeSalt*Salt;    CDOM effect related to Salinity
    !-----------------------------------------------------------------------
    iKe = 0

    Ke0    = 0.26            !backgroud light extinction coefficient (1/m)
    KeC    = 0.017           !Light attenu. due to chlorophyll
    KeS    = 0.07            !Light attenu. due to TSS
    KeSalt = -0.02           !Light attenu. due to CDOM (related to salinity)
    tss2c  = 6.0             !TSS to carbon ratio

    !-----------------------------------------------------------------------
    !options of computing light limitation factor
    !  iLight=0: (Carl Cerco), unit: E/m^2
    !  iLight=1: todo (add other options)
    !-----------------------------------------------------------------------
    iLight = 0

    alpha  = 8.0   8.0  8.0  !init. slope of P-I curve (g[C]*m2/g[Chl]/E), (iLight=0)

    !-----------------------------------------------------------------------
    !options of phytoplankton' predation term
    !-----------------------------------------------------------------------
    iPR = 1   !(0: linear formulation; 1: quadratic)

    PRR = 0.1  0.2  0.05   !predation rate by higher trophic level (day-1, or day-1.g-1.m3)

    !-----------------------------------------------------------------------
    !ICM init. value (wqc0), settling velocity (WSP), net settling velocity (WSPn) (m.day-1)
    !-----------------------------------------------------------------------
    !name: (PB1,PB2,PB3)   (RPOC,LPOC,DOC)   (RPON,LPON,DON,NH4,NO3)     (RPOP,LPOP,DOP,PO4)    (COD,DO)
    wqc0 = 1.0  0.5  0.05  1.0  0.5  0.5     0.15 0.15 0.05 0.01 0.05   0.005 0.005 0.01 0.05   0.0 12.0
    WSP  = 0.3  0.10 0.0   0.25 0.25 0.0     0.25 0.25 0.0  0.0  0.0    0.25  0.25  0.0  1.0    0.0 0.0
    WSPn = 0.3  0.10 0.0   0.25 0.25 0.0     0.25 0.25 0.0  0.0  0.0    0.25  0.25  0.0  1.0    0.0 0.0

    !-----------------------------------------------------------------------
    !Silica model (0: OFF;  1: ON)
    !-----------------------------------------------------------------------
    iSilica = 0

    !-----------------------------------------------------------------------
    !Zooplankton(iZB=1: use zooplankton dynamics; iZB=0: don't use)
    !-----------------------------------------------------------------------
    iZB = 0

    !-----------------------------------------------------------------------
    !PH model (0: OFF;  1: ON)
    !-----------------------------------------------------------------------
    iPh = 0

    !-----------------------------------------------------------------------
    !ChesBay Program Model (0: OFF;  1: ON)
    !-----------------------------------------------------------------------
    iCBP = 0

    !-----------------------------------------------------------------------
    !Submerged Aquatic Vegetation switch
    !-----------------------------------------------------------------------
    isav_icm = 0

    !-----------------------------------------------------------------------
    !Intertidal vegetation switch
    !-----------------------------------------------------------------------
    iveg_icm = 0                   !0: VEG off;  1: VEG on

    !-----------------------------------------------------------------------
    !Sediment module switch. iSed=1: Use sediment flux model
    !benthic flux provided by sed_flux_model
    !-----------------------------------------------------------------------
    iSed = 1

    !-----------------------------------------------------------------------
    !Benthic Algae
    !-----------------------------------------------------------------------
    iBA = 0

    !-----------------------------------------------------------------------
    !solar radiation: dim=(npt,time) or (time,)
    !  iRad=0: short wave from sflux; require ihconsv =1, nws=2
    !  iRad=1: short wave from ICM_rad.th.nc (unit: E.m-2.day-1)
    !  note: npt=1/np/ne/other; need to add "elements" number if npt=other
    !-----------------------------------------------------------------------
    iRad = 0

    !-----------------------------------------------------------------------
    !Atmospheric and Bottom fluxes: dim=(npt,ntrs_icm,time)
    !  isflux/ibflux=1: additional nutrient fluxes from ICM_sflux.th.nc/ICM_bflux.th.nc
    !  units: g/m2,(consistent with variable units in the model)
    !  note: npt=1/np/ne/other; need to add "elements" number if npt=other
    !-----------------------------------------------------------------------
    isflux = 0
    ibflux = 0

    !-----------------------------------------------------------------------
    !ICM station outputs (need istation.in with *.bp format)
    !-----------------------------------------------------------------------
    iout_icm = 0
    nspool_icm = 24

    !-----------------------------------------------------------------------
    !options of nutrient limitation on phytoplankton growth
    !  iLimit=0: f=min[f(N),f(P)]*f(I);  iLimit=1: f=min[f(N),f(P),f(I)]
    !-----------------------------------------------------------------------
    iLimit   = 0

    !-----------------------------------------------------------------------
    !idry_icm=1: turn on shallow kinetic biochemical process
    !idry_icm=0: jump dry elems, keep last wet value; jump won't happen if iveg_icm is turn on and this elem is veg
    !-----------------------------------------------------------------------
    idry_icm = 0
    /

    &CORE
    !-----------------------------------------------------------------------
    !ICM parameters for water column; PB: phytoplankton
    !CBP module parameters are included: (SRPOC,SRPON,SRPOP,PIP)
    !-----------------------------------------------------------------------
    !phytoplankton growth
    GPM  = 2.5    2.8    3.5        !PB growth rates (day-1)
    TGP  = 15.0   22.0   27.0        !optimal temp. for PB growth (oC)
    KTGP = 0.005  0.004  0.003    0.008  0.006  0.004  !temp. dependence for PB growth; dim(PB=1:3,1:2) (oC-2)

    !phytoplankton photorespiration & metabolism
    MTR  = 0.0    0.0    0.0          !PB photorespiration coefficient (0<MTR<1)
    MTB  = 0.01   0.02   0.03         !PB metabolism rates (day-1)
    TMT  = 20.0   20.0   20.0         !reference temp. for PB metabolism (oC)
    KTMT = 0.0322 0.0322 0.0322       !temp. dependence for PB metabolism (oC-1)

    !partition of phytoplankton biomass (predation)
    FCP  = 0.35 0.30 0.20  0.55 0.50 0.50  0.10 0.20 0.30  0.0  0.0  0.0 !fractions of PB carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,3)
    FNP  = 0.35 0.35 0.35  0.50 0.50 0.50  0.10 0.10 0.10  0.05 0.05 0.05  0.0 0.0 0.0  !fractions of PB nitrogen into (RPON,LPON,DON,NH4,SRPON)
    FPP  = 0.10 0.10 0.10  0.20 0.20 0.20  0.50 0.50 0.50  0.20 0.20 0.20  0.0 0.0 0.0  !fractions of PB Phosphorus into (RPOP,LPOP,DOP,PO4,SRPOP)

    !partition of metabolized phytoplankton biomass
    FCM  = 0.0 0.0 0.0   0.0 0.0 0.0   0.1 0.1 0.1   0.0 0.0 0.0 !fractions of PB metabolism carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,4)
    FNM  = 0.0 0.0 0.0   0.0 0.0 0.0   1.0 1.0 1.0   0.0 0.0 0.0  0.0 0.0 0.0 !fractions of PB metabolism N. into (RPON,LPON,DON,NH4,SRPON); dim(PB=1:3,5)
    FPM  = 0.0 0.0 0.0   0.0 0.0 0.0   1.0 1.0 1.0   0.0 0.0 0.0  0.0 0.0 0.0 !fractions of PB metabolism P. into (RPOP,LPOP,DOP,PO4,SRPOP); dim(PB=1:3,5)

    !nitrification
    Nit    = 0.07                     !maximum nitrification rate (day-1)
    TNit   = 27.0                     !optimal temp. for nitrification (oC)
    KTNit  = 0.0045  0.0045           !temp. dependence (T<=TNit & T>TNit) for nitrification (oC-2)
    KhDOn  = 1.0                      !DO half saturation for nitrification (mg/L)

    !denitrifcation
    KhDOox  = 0.5                     !DO half saturation for dentrification & DOC's oxic respiration (mg/L)
    KhNO3dn = 0.1                     !NO3 half saturation for denitrification (mg/L)

    !decay coefficients for organic matters (C,N,P); CBP module: (KSR,TRSR,KTRSR,KPIP)
    KC0   = 0.005  0.075  0.2         !minimum decay rate of RPOC,LPOC,DOC (day-1)
    KN0   = 0.005  0.075  0.2         !minimum decay rate of RPON,LPON,DON (day-1)
    KP0   = 0.005  0.075  0.2         !minimum decay rate of RPOP,LPOP,DOP (day-1)
    KCalg = 0.0    0.0    0.0         !algae effect on RPOC,LPOC,DOC decay (day-1.m3.g[C]-1)
    KNalg = 0.0    0.0    0.0         !algae effect on RPON,LPON,DON decay (day-1.m3.g[C]-1)
    KPalg = 0.0    0.0    0.0         !algae effect on RPOP,LPOP,DOP decay (day-1.m3.g[C]-1)
    TRM   = 20.0   20.0   20.0        !reference temp. for (RPOM,LPOM,DOM) decay (oC)
    KTRM  = 0.069  0.069  0.069       !temp. dependence for (RPOM,LPOM,DOM) decay (oC-1)
    KSR0  = 0.001  0.001  0.001       !decay rates of SRPOC,SRPON,SRPOP (day-1)
    TRSR  = 20.0   20.0   20.0        !reference temp. for (SRPOC,SRPON,SRPOP) decay (oC)
    KTRSR = 0.069  0.069  0.069       !temp. dependence for (SRPOC,SRPON,SRPOP) decay (oC-1)
    KPIP  = 0.0                       !dissolution rate of PIP (day-1)

    !decay coefficient for chemical oxygen demand (COD)
    KCD   = 1.0                       !oxidation rate of COD at TRCOD (day-1)
    TRCOD = 20.0                      !reference temp. for COD oxidation (oC)
    KTRCOD= 0.041                     !temp. dependence for COD oxidation (oC-1)
    KhCOD=  1.5                       !COD half saturation for COD oxidation (mg[O2]/L)

    !nutrient limitation & salinity stress
    KhN  = 0.01   0.01   0.01         !nitrogen half saturation (mg/L)
    KhP  = 0.001  0.001  0.001        !phosphorus half saturation (mg/L)
    KhSal= 1e6    1e6    0.1          !salinity when PB growth is halved (PSU); (1e6: no salinity stress)

    !conversion coeffiecients
    c2chl= 0.059 0.059 0.059          !carbon to chlorophyll ratio (g[C]/mg[Chl])
    n2c  = 0.167 0.167 0.167          !nitrogen to carbon ratio for phytoplankton
    p2c  = 0.02  0.02  0.02           !phosphorus to carbon ratio for phytoplankton
    o2c  = 2.67                       !oxygen to carbon ratio in respiration
    o2n  = 4.33                       !oxygen to ammonium ratio (g[O2]/g[NH4])
    dn2c = 0.933                      !mass of NO3 consumed per mass of DOC oxidized in denit. (g[N]/g[C])
    an2c = 0.5                        !ratio of denit. rate to oxic DOC respiration rate

    !misc
    KhDO =   0.5   0.5   0.5          !DO half saturation for PB's DOC excretion (mg/L)
    KPO4p=   0.0                      !coefficient relating PO4 sorption to TSS
    WRea =   0.0                      !baseline wind-induced reaeration coefficient for DO (day-1)
    PBmin=   0.01  0.01  0.01         !minimum PB concentration (mg[C]/L)
    dz_flux= 1.0   1.0                !surface/bottom thikness (m) within which sflux/bflux are redistributed
    /

    &SFM
    !-----------------------------------------------------------------------
    ! sediment flux model (SFM) parameter
    !-----------------------------------------------------------------------
    !initial sediment concentrations in lower layer (g/m3); For cold start
    btemp0 = 5.0     !initial temp. (oC)
    bstc0  = 0.1     !surface transfer coefficient
    bSTR0  = 0.0     !benthic stress (day)
    bThp0  = 0.0     !consective days of hypoxia (day)
    bTox0  = 0.0     !consective days of oxic condition after hypoxia event (day)
    bNH40  = 4.0     !NH4
    bNO30  = 1.0     !NO3
    bPO40  = 5.0     !PO4
    bH2S0  = 250.0   !H2S
    bCH40  = 40.0    !CH4
    bPOS0  = 500.0   !POS
    bSA0   = 500.0   !SA
    bPOC0  = 1000.0  3000.0   5000.0    !POC(G=1:3)
    bPON0  = 150.0   500.0    1500.0    !PON(G=1:3)
    bPOP0  = 30.0    300.0    500.0     !POP(G=1:3)

    !basic information about sediment
    bdz    = 0.1         !sediment thickness (m)
    bVb    = 1.37e-5     !burial rate (m.day-1); 1 cm/yr=2.74e-5 m.day-1
    bsolid = 0.5  0.5    !sediment solid conc. in Layer 1 and Layer 2 (Kg.L-1)
    bdiff  = 1.8e-7      !diffusion coefficient for sediment temp. (m2/s)
    bTR    = 20          !reference temp. for sediment processes

    !particle mixing and diffusion coefficients, benthic stress
    bVpmin   = 3.0e-6   !minimum particle mixing velocity coefficient (m.day-1)
    bVp      = 1.2e-4   !particle mixing velocity coefficient (m.day-1)
    bVd      = 1.0e-3   !diffusion velocity coefficient (m.day-1)
    bKTVp    = 1.117    !temp. dependece of particle mixing velocity
    bKTVd    = 1.08     !temp. dependece of diffusion velocity
    bKST     = 0.03     !1st order decay rate of benthic stress  (day-1)
    bSTmax   = 20.0     !maximum value of benthic stress (day) (note: smaller than 1/bKST)
    bKhDO_Vp = 4.0      !DO half-saturation of particle mixing (mg/L)
    bDOc_ST  = 1.0      !DO criteria for benthic stress (mg/L)
    banoxic  = 10.0     !consective days of hypoxia causing maximum benthic stress (day)
    boxic    = 45.0     !time lag for bethos recovery from hypoxia event (day)
    bp2d     = 0.0      !ratio from mixing coef. to diffusion coef. (benthos enhanced effect)

    !decay rates of sediment POM (diagenesis flux)
    bKC    = 0.035   0.0018   0.00     !decay rate of POC (3G class) at bTR (day-1)
    bKN    = 0.035   0.0018   0.00     !decay rate of PON (3G class) at bTR (day-1)
    bKP    = 0.035   0.0018   0.00     !decay rate of POP (3G class) at bTR (day-1)
    bKTC   = 1.10    1.150    1.17     !temp. dependence of POC decay (oC-1)
    bKTN   = 1.10    1.150    1.17     !temp. dependence of PON decay (oC-1)
    bKTP   = 1.10    1.150    1.17     !temp. dependence of POP decay (oC-1)

    !fraction of phytoplankton into sediment 3G POM  (depositional flux)
    bFCP = 0.35  0.55  0.01    0.35  0.55  0.01    0.35  0.55  0.01    !Phyto POC into sed POC (G3,PB=1:3)
    bFNP = 0.35  0.55  0.01    0.35  0.55  0.01    0.35  0.55  0.01    !Phyto PON into sed PON (G3,PB=1:3)
    bFPP = 0.35  0.55  0.01    0.35  0.55  0.01    0.35  0.55  0.01    !Phyto POP into sed POP (G3,PB=1:3)

    !fraction of RPOM into sediment 3G POM (depositional flux, for iCBP=0); when iCBP=1, RPOM will be routed to sediment G2 pool
    bFCM = 0.0   0.43   0.57          !refractory POC into sed POC(3G)
    bFNM = 0.0   0.54   0.46          !refractory PON into sed PON(3G)
    bFPM = 0.0   0.43   0.57          !refractory POP into sed POP(3G)

    !nitrification (NH4)
    bKNH4f    = 0.20    !NH4 reaction rate in freshwater  at bTR (1st layer) (m/day)
    bKNH4s    = 0.140   !NH4 reaction rate in salty water at bTR (1st layer) (m/day)
    bKTNH4    = 1.08    !temp. dependency for NH4 reaction (oC-1)
    bKhNH4    = 1.5     !half-stauration NH4 for nitrification (g/m3)
    bKhDO_NH4 = 2.00    !half-stauration DO for nitrification (g/m3)
    bpieNH4   = 1.0     !partition coefficients of NH4 in Layer 1 & 2 (Kg-1.L)
    bsaltn    = 1.0     !salinity criteria of fresh/salty water for NH4/NO3 reaction (PSU)

    !denitrification (NO3)
    bKNO3f    = 0.30    !NO3 reaction rate in freshwater at bTR (1st layer) (m/day)
    bKNO3s    = 0.125   !NO3 reaction rate in salty water at bTR (1st layer) (m/day)
    bKNO3     = 0.25    !NO3 reaction rate (2nd layer) (m/day)
    bKTNO3    = 1.08    !temp. dependency for NO3 reaction (oC-1)

    !sulfide oxidation (H2S)
    bKH2Sd    = 0.2     !dissolved H2S reaction rate at bTR (1st layer) (m/day)
    bKH2Sp    = 0.4     !particulate H2S reaction rate at bTR (1st layer) (m/day)
    bKTH2S    = 1.08    !temp. dependency for H2S reaction (oC-1)
    bpieH2Ss  = 100.0   !partition coefficient of NH4 in Layer 1 (Kg-1.L)
    bpieH2Sb  = 100.0   !partition coefficient of NH4 in Layer 2 (Kg-1.L)
    bKhDO_H2S = 8.0     !O2 constant to normalize H2S oxidation (g[O2]/m3)
    bsaltc    = 1.0     !salinity criteria of fresh/salty water for carbon reaction (PSU)

    !methane oxidation (CH4)
    bKCH4     = 0.2     !CH4 reaction rate at bTR (1st layer) (m/day)
    bKTCH4    = 1.08    !temp. dependency for CH4 reaction
    bKhDO_CH4 = 0.2     !half-saturation DO for CH4 oxidation (g[O2]/m3)
    bo2n      = 2.86    !oxygen to nitrogen ratio in sediment (denitrification)

    !phosphate reaction (PO4)
    bpiePO4   = 50.0    !partition coefficient of PO4 in Layer 2 (Kg-1.L)
    bKOPO4f   = 3000.0  !oxygen dependency for PO4 sorption in freshwater in Layer 1
    bKOPO4s   = 300.0   !oxygen dependency for PO4 sorption in salty water in Layer 1
    bDOc_PO4  = 1.0     !DO criteria for PO4 sorptiona (g[O2]/m3)
    bsaltp    = 1.0     !salinity criteria of fresh/salty water for PO4 partition (PSU)

    !silica dissolution (SI)
    bKS       = 0.50    !decay rate of POS (3G class) at bTR
    bKTS      = 1.10    !temp. dependence of POS decay
    bSIsat    = 40.0    !silica saturation conc. in pore water (g[Si]/m3)
    bpieSI    = 100.0   !partition coefficient of silica in Layer 2 (Kg-1.L)
    bKOSI     = 10.0    !oxygen dependency for silica sorption in Layer 1
    bKhPOS    = 5.0e4   !POS half saturation for POS dissolution (g/m3)
    bDOc_SI   = 1.0     !DO criteria for silica sorptiona (g[O2]/m3)
    bJPOSa    = 0.0     !additional POS flux associated with POM detrius beside algea (g.m-2.day-1)

    !SAV and VEG contribution to sediment POM (for isav_icm=1, iveg_icm=1)
    bFCs = 0.65  0.255  0.095         !SAV POC into 3G sed 3G POC
    bFNs = 0.65  0.300  0.050         !SAV PON into 3G sed 3G PON
    bFPs = 0.65  0.255  0.095         !SAV POP into 3G sed 3G POP
    bFCv = 0.65  0.255  0.095   0.65  0.255  0.095   0.65  0.255  0.095   !VEG POC into sed POC (G3,PB=1:3)
    bFNv = 0.65  0.300  0.050   0.65  0.300  0.050   0.65  0.300  0.050   !VEG PON into sed PON (G3,PB=1:3)
    bFPv = 0.65  0.255  0.095   0.65  0.255  0.095   0.65  0.255  0.095   !VEG POP into sed POP (G3,PB=1:3)
    /

    &Silica
    !-----------------------------------------------------------------------
    !Silica parameters
    !-----------------------------------------------------------------------
    !silica
    FSP  = 0.90   0.10                !fractions of diatom silica into (SU,SA)
    FSM  = 0.50   0.50                !fractions of diatom metabolism Si into (SU,SA)
    KS   = 0.03                       !dissolution rate of SU at TRS (day-1)
    TRS  = 20.0                       !reference temp. for SU dissolution (oC)
    KTRS = 0.092                      !temp. dependence for SU dissolution (oC-1)
    KhS  = 0.05   0.0   0.0           !silica half saturation (mg/L); (0.0: no Si limitation)
    s2c  = 0.50   0.0   0.0           !silica to carbon ratio for phytolankton; (0.0: no Si uptake)
    KSAp = 0.0                        !coefficient relating Silicate(SA) sorption to TSS
    /

    &ZB
    !-----------------------------------------------------------------------
    !Zooplankton (ZB) parameters
    !ZB prey(1:8)=(ZB1,ZB2,PB1,PB2,PB3,RPOC,LPOC,DOC)
    !-----------------------------------------------------------------------
    !zooplankton growth
    zGPM = 0.0   0.0   1.75  1.75  1.75  1.75  1.75  1.75    1.0   0.0   2.0   2.0   2.0   2.0   2.0   2.0   !ZB predation rate(day-1); dim(prey=1:8, ZB=1:2)
    zKhG = 0.175 0.175 0.175 0.175 0.175 0.175 0.175 0.175   0.175 0.175 0.175 0.175 0.175 0.175 0.175 0.175 !reference prey conc.(mg/L); dim(prey=1:8, ZB=1:2)
    zTGP = 25.0   25.0               !optimal temp. for ZB growth (oC)
    zKTGP= 0.0035 0.008  0.025 0.030 !temp. dependence for ZB growth (T<=zTGP & T>zTGP); dim(ZB=1:2,1:2) (oC-2)
    zAG  = 0.75                      !ZB assimilation efficiency ratio (0-1)
    zRG  = 0.1                       !ZB respiration ratio when it grazes (0-1)

    !zooplankton mortality & metabolism
    zMRT  = 0.02   0.02              !ZB mortality rates (day-1)
    zMTB  = 0.254  0.186             !ZB metabolism rates (day-1)
    zTMT  = 20.0   20.0              !reference temp. for ZB metabolism (oC)
    zKTMT = 0.0693 0.0693            !temp. dependence for ZB metabolism (oC-1)

    !partition of zooplankton biomass
    zFCP = 0.35   0.55   0.10        !fractions of ZB carbon into (RPOC,LPOC,DOC)
    zFNP = 0.35   0.50   0.10  0.05  !fractions of ZB nitrogen into (RPON,LPON,DON,NH4)
    zFPP = 0.10   0.20   0.50  0.20  !fractions of ZB Phosphorus into (RPOP,LPOP,DOP,PO4)
    zFSP = 0.70   0.25               !fractions of ZB silica into (SU,SA)

    !partition of metabolized zooplankton biomass
    zFCM = 0.10   0.0                !fractions of ZB metabolism carbon into DOC; dim(ZB=1:2)
    zFNM = 0.35 0.30   0.50 0.40   0.10 0.20   0.05 0.10 !fractions of ZB metabolism N. into (RPON,LPON,DON,NH4); dim(ZB=1:2,4)
    zFPM = 0.35 0.30   0.50 0.40   0.10 0.20   0.05 0.10 !fractions of ZB metabolism P. into (RPOP,LPOP,DOP,PO4); dim(ZB=1:2,4)
    zFSM = 0.50 0.40   0.50 0.60     !fractions of ZB metabolism Si into (SU,SA); dim(ZB=1:2,2)

    !misc
    zKhDO= 0.5   0.5                  !DO half saturation for ZB's DOC excretion (mg/L)
    zn2c = 0.20  0.20                 !nitrogen to carbon ratio for zooplankton
    zp2c = 0.02  0.02                 !phosphorus to carbon ratio for zooplankton
    zs2c = 0.50  0.50                 !silica to carbon ratio for zooplankton
    z2pr = 0.5   0.5                  !ratio converting ZB+PB biomass to predation rates on ZB (L.mg-1.day-1)
    p2pr = 0.25                       !ratio converting ZB+PB biomass to predation rates on PB (L.mg-1.day-1)
    /

    &PH_ICM
    !-----------------------------------------------------------------------
    !PH model parameter
    !-----------------------------------------------------------------------
    ppatch0  = -999               !region flag for pH (1: ON all elem.; -999: spatial)
    pKCACO3  = 60.0               !dissulution bewteen CaCO3 and Ca++
    pKCA     = 60.0               !sediment surface transfer coefficient from CaCO3 to Ca++
    pRea     = 1.0                !reaeration rate for CO2
    inu_ph   = 0                  !nudge option for pH model
    /

    &SAV
    !-----------------------------------------------------------------------
    !Submerged Aquatic Vegetation (SAV) parameters
    !-----------------------------------------------------------------------
    spatch0 = -999              !region flag for SAV. (1: ON all elem.; -999: spatial)
    stleaf0 = -999              !init conc of total sav leaf
    ststem0 = -999              !init conc of total sav stem
    stroot0 = -999              !init conc of total sav root

    !growth coefficients
    sGPM  = 0.1                 !maximum growth rate (day-1)
    sTGP  = 32                  !optimal growth temperature (oC)
    sKTGP = 0.003  0.005        !temp. dependence for growth (T<=sTGP & T>sTGP)
    sFAM  = 0.2                 !fraction of leaf production to active metabolism
    sFCP  = 0.6    0.3    0.1   !fractions of production to leaf/stem/root biomass

    !metabolism coefficients
    sMTB  = 0.02   0.02   0.02  !metabolism rates of leaf/stem/root
    sTMT  = 20     20     20    !reference temp. for leaf/stem/root metabolism
    sKTMT = 0.069  0.069  0.069 !temp. dependence of leaf/stem/root metabolism
    sFCM  = 0.05   0.15   0.3   0.5  !fractions of metabolism C into (RPOC,LPOC,DOC,CO2)
    sFNM  = 0.05   0.15   0.3   0.5  !fractions of metabolism N into (RPON,LPON,DON,NH4)
    sFPM  = 0.05   0.1    0.35  0.5  !fractions of metabolism P into (RPOP,LPOP,DOP,PO4)

    !nutrient limitation
    sKhNw  = 0.01               !nitrogen half saturation in water column
    sKhNs  = 0.1                !nitrogen half saturation in sediments
    sKhNH4 = 0.1                !ammonium half saturation
    sKhPw  = 0.001              !phosphorus half saturation in water column
    sKhPs  = 0.01               !phosphorus half saturation in sediments

    !misc. coefficients
    salpha = 0.006              !init. slope of P-I curve (g[C]*m2/g[Chl]/E)
    sKe    = 0.045              !light attenuation due to sav absorption
    shtm   = 0.054  2.0         !minimum (base) and  maximium canopy height
    s2ht   = 0.0036 0.0036 0.0  !coeffs. converting (leaf,stem,root) to canopy height
    sc2dw  = 0.38               !carbon to dry weight ratio of SAV
    s2den  = 10                 !coeff. computing SAV density from leaf&stem
    sn2c   = 0.09               !nitrogen to carbon ratio of sav
    sp2c   = 0.01               !phosphorus to carbon ratio
    so2c   = 2.67               !oxygen to carbon ratio
    /

    &VEG
    !-----------------------------------------------------------------------
    !Intertidal Vegetation (VEG), paramters
    !-----------------------------------------------------------------------
    vpatch0 = -999                 !region flag for VEG. (1: ON all elem.; -999: spatial)
    vtleaf0 = 100.0  100.0  100.0  !init conc to total veg leaf
    vtstem0 = 100.0  100.0  100.0  !init conc to total veg stem
    vtroot0 = 30.0   30.0   30.0   !init conc to total veg root

    !growth coefficients: see description of variable meanings above
    vGPM  = 0.1    0.1    0.1      !maximum growth rate (day-1)
    vFAM  = 0.2    0.2    0.2      !fractions of leaf production to active metabolism
    vTGP  = 32.0   32.0   32.0     !optimal growth temperature (oC)
    vKTGP = 0.003  0.003  0.003   0.005  0.005  0.005   !temp. dependence(3,2) for growth (T<=vTGP & T>vTGP)
    vFCP  = 0.6    0.6    0.6     0.3    0.3    0.3     0.1   0.1  0.1  !fractions of production to leaf/stem/root biomass; dim(veg,leaf/stem/root)

    !metabolism coefficients: see description of variable meanings above
    vMTB  = 0.02  0.02  0.02    0.02  0.02  0.02    0.01  0.01  0.01  !metabolism rates of leaf/stem/root; dim(veg,leaf/stem/root)
    vTMT  = 20.0  20.0  20.0    20.0  20.0  20.0    20.0  20.0  20.0  !reference temp. for leaf/stem/root metabolism
    vKTMT = 0.069 0.069 0.069   0.069 0.069 0.069   0.069 0.069 0.069 !temp. depdenpence for leaf/stem/root metabolism
    vFNM  = 0.05  0.05  0.05    0.15  0.15  0.15    0.3  0.3  0.3    0.5 0.5 0.5 !fractions(3,4) of metabolism N into (RPON,LPON,DON,NH4)
    vFPM  = 0.05  0.05  0.05    0.1   0.1   0.1     0.35 0.35 0.35   0.5 0.5 0.5 !fractions(3,4) of metabolism P into (RPOP,LPOP,DOP,PO4)
    vFCM  = 0.05  0.05  0.05    0.15  0.15  0.15    0.3  0.3  0.3    0.5 0.5 0.5 !fractions(3,4) of metabolism N into (RPOC,LPOC,DOC,CO2)
    ivNc  = 1                   !recycled veg N goes to (0: sediment; 1: water)
    ivPc  = 1                   !recycled veg P goes to (0: sediment; 1: water)

    !nutrient limitation & salinity/inundation stress
    vKhNs = 0.1   0.1   0.1     !nitrogen half saturation in sediments
    vKhPs = 0.01  0.01  0.01    !phosphorus half saturation in sediments
    vScr  = 35.0  35.0  35.0    !reference sality for computing veg growth
    vSopt = 35.0  15.0  0.0     !optimal salinity for veg growth
    vInun = 1.0   1.0   1.0     !reference value for inundation stress (nondimensional)
    ivNs  = 1                   !N limitation on veg growth(0: OFF; 1: ON)
    ivPs  = 1                   !P limitation on veg growth(0: OFF; 1: ON)

    !mortality (seasonal?)
    ivMRT = 0                    !veg mortality term (0: OFF;  1: ON)
    vTMR  = 17.0   17.0  17.0    17.0  17.0  17.0  !reference temp(3,2) for leaf/stem mortality
    vKTMR = 4.0    4.0   4.0     4.0   4.0   4.0   !temp dependence(3,2) for leaf/stem mortality
    vMR0  = 12.8   12.8  12.8    12.8  12.8  12.8  !base value(3,2) of temp effect on mortality (unit: None)
    vMRcr = 15.0   15.0  15.0    15.0  15.0  15.0  !reference value(3,2) for computing mortality (unit: None)

    !misc. coefficients
    valpha = 0.006  0.006  0.006  !init. slope of P-I curve
    vKe    = 0.045  0.045  0.045  !light attenuation from veg absorption
    vht0   = 0.054  0.054  0.054  !base veg canopy hgt (vht)
    vcrit  = 250.0  250.0  250.0  !critical mass for computing vht
    v2ht   = 0.0036 0.0036 0.0036 0.001 0.001 0.001   !coefs. convert mass(3,2) to canopy height
    vc2dw  = 0.38   0.38   0.38   !carbon to dry weight ratio of VEG
    v2den  = 10     10     10     !coeff. computing veg density
    vp2c   = 0.01   0.01   0.01   !phosphorus to carbon ratio
    vn2c   = 0.09   0.09   0.09   !nitrogen to carbon ratio
    vo2c   = 2.67   2.67   2.67   !oxygen to carbon ratio
    /

    &BAG
    !-----------------------------------------------------------------------
    !Benthic Algae
    !-----------------------------------------------------------------------
    gpatch0 =  -999          !region flag for BA. (1: ON all elem.; -999: spatial)
    BA0     =  5.0           !initial BA concentration (g[C].m-2)
    gGPM    =  2.25          !BA maximum growth rate (day-1)
    gTGP    =  20.0          !optimal temp. for BA growth (oC)
    gKTGP   =  0.004  0.006  !temp. dependence for BA growth (oC-2)
    gMTB    =  0.05          !respiration rate at temp. of gTR (day-1)
    gPRR    =  0.1           !predation rate at temp. of gTR (day-1)
    gTR     =  20.0          !reference temperature for BA respiration (oC)
    gKTR    =  0.069         !temp. dependence for BA respiration (oC-1)
    galpha  =  0.1           !init. slope of P-I curve (m2/E)
    gKSED   =  0.0           !light attenuation due to sediment (None)
    gKBA    =  0.01          !light attenuation coef. due to BA self-shading (g[C]-1.m2)
    gKhN    =  0.01          !nitrogen half saturation for BA growth (g[N]/m2)
    gKhP    =  0.001         !phosphorus half saturation for BA growth (g[P]/2)
    gp2c    = 0.0167         !phosphorus to carbon ratio
    gn2c    = 0.167          !nitrogen to carbon ratio
    go2c    = 2.67           !oxygen to carbon ratio
    gFCP    = 0.5 0.45 0.05  !fraction of predation BA C into 3G classes in sediment
    gFNP    = 0.5 0.45 0.05  !fraction of predation BA N into 3G classes in sediment
    gFPP    = 0.5 0.45 0.05  !fraction of predation BA P into 3G classes in sediment
    /

    &ERO
    !-----------------------------------------------------------------------
    ! benthic erosion parameter
    !-----------------------------------------------------------------------
    ierosion = 0         !1: H2S flux; 2: POC flux; 3:  H2S&POC flux
    erosion  = 864       !erosion rate (kg.m-2.day)
    etau     = 1.e-6     !critical bottom shear stress (Pa)
    eporo    = 0.8       !coefficinet in erosion formula (see code in icm_sfm.F90)
    efrac    = 0.5       !fraction coefficinet in erosion formula (see code in icm_sfm.F90)
    ediso    = 2.5       !H2S erosion coeffcient (see code in icm_sfm.F90)
    dfrac    = 0.02 0.02 !deposition fraction of POC (negative value: dfrac will be computed)
    dWS_POC  = 3.0  3.0  !coefficient in POC erosion (see code in icm_sfm.F90)
    /

    """

    marco: Optional[Marco] = Field(default=None)
    core: Optional[Core] = Field(default=None)
    sfm: Optional[Sfm] = Field(default=None)
    silica: Optional[Silica] = Field(default=None)
    zb: Optional[Zb] = Field(default=None)
    ph_icm: Optional[Ph_icm] = Field(default=None)
    sav: Optional[Sav] = Field(default=None)
    stem: Optional[Stem] = Field(default=None)
    veg: Optional[Veg] = Field(default=None)
    bag: Optional[Bag] = Field(default=None)
    ero: Optional[Ero] = Field(default=None)
    poc: Optional[Poc] = Field(default=None)
