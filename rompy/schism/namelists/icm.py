from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel

class MARCO(NamelistBaseModel):
    nsub: int = Field(1, description="")
    iKe: int = Field(0, description="")
    Ke0: float = Field(0.26, description="backgroud light extinction coefficient (1/m)")
    KeC: float = Field(0.017, description="Light attenu. due to chlorophyll")
    KeS: float = Field(0.07, description="Light attenu. due to TSS")
    KeSalt: float = Field(-0.02, description="Light attenu. due to CDOM (related to salinity)")
    tss2c: float = Field(6.0, description="TSS to carbon ratio")
    iLight: int = Field(0, description="")
    alpha: list = Field([8.0, 8.0, 8.0], description="init. slope of P-I curve (g[C]*m2/g[Chl]/E), (iLight=0)")
    iPR: int = Field(1, description="(0: linear formulation; 1: quadratic)")
    PRR: list = Field([0.1, 0.2, 0.05], description="predation rate by higher trophic level (day-1, or day-1.g-1.m3)")
    wqc0: list = Field([1.0, 0.5, 0.05, 1.0, 0.5, 0.5, 0.15, 0.15, 0.05, 0.01, 0.05, 0.005, 0.005, 0.01, 0.05, 0.0, 12.0], description="")
    WSP: list = Field([0.3, 0.1, 0.0, 0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0], description="")
    WSPn: list = Field([0.3, 0.1, 0.0, 0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0], description="")
    iSilica: int = Field(0, description="")
    iZB: int = Field(0, description="")
    iPh: int = Field(0, description="")
    iCBP: int = Field(0, description="")
    isav_icm: int = Field(0, description="")
    iveg_icm: int = Field(0, description="0: VEG off;  1: VEG on")
    iSed: int = Field(1, description="")
    iBA: int = Field(0, description="")
    iRad: int = Field(0, description="")
    isflux: int = Field(0, description="")
    ibflux: int = Field(0, description="")
    iout_icm: int = Field(0, description="")
    nspool_icm: int = Field(24, description="")
    iLimit: int = Field(0, description="")
    idry_icm: int = Field(0, description="")

class CORE(NamelistBaseModel):
    GPM: list = Field([2.5, 2.8, 3.5], description="PB growth rates (day-1)")
    TGP: list = Field([15.0, 22.0, 27.0], description="optimal temp. for PB growth (oC)")
    KTGP: list = Field([0.005, 0.004, 0.003, 0.008, 0.006, 0.004], description="temp. dependence for PB growth; dim(PB=1:3,1:2) (oC-2)")
    MTR: list = Field([0.0, 0.0, 0.0], description="PB photorespiration coefficient (0<MTR<1)")
    MTB: list = Field([0.01, 0.02, 0.03], description="PB metabolism rates (day-1)")
    TMT: list = Field([20.0, 20.0, 20.0], description="reference temp. for PB metabolism (oC)")
    KTMT: list = Field([0.0322, 0.0322, 0.0322], description="temp. dependence for PB metabolism (oC-1)")
    FCP: list = Field([0.35, 0.3, 0.2, 0.55, 0.5, 0.5, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0], description="fractions of PB carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,3)")
    FNP: list = Field([0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0], description="fractions of PB nitrogen into (RPON,LPON,DON,NH4,SRPON)")
    FPP: list = Field([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0], description="fractions of PB Phosphorus into (RPOP,LPOP,DOP,PO4,SRPOP)")
    FCM: list = Field([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0], description="fractions of PB metabolism carbon into (RPOC,LPOC,DOC,SRPOC); dim(PB=1:3,4)")
    FNM: list = Field([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], description="fractions of PB metabolism N. into (RPON,LPON,DON,NH4,SRPON); dim(PB=1:3,5)")
    FPM: list = Field([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], description="fractions of PB metabolism P. into (RPOP,LPOP,DOP,PO4,SRPOP); dim(PB=1:3,5)")
    Nit: float = Field(0.07, description="maximum nitrification rate (day-1)")
    TNit: float = Field(27.0, description="optimal temp. for nitrification (oC)")
    KTNit: list = Field([0.0045, 0.0045], description="temp. dependence (T<=TNit & T>TNit) for nitrification (oC-2)")
    KhDOn: float = Field(1.0, description="DO half saturation for nitrification (mg/L)")
    KhDOox: float = Field(0.5, description="DO half saturation for dentrification & DOC's oxic respiration (mg/L)")
    KhNO3dn: float = Field(0.1, description="NO3 half saturation for denitrification (mg/L)")
    KC0: list = Field([0.005, 0.075, 0.2], description="minimum decay rate of RPOC,LPOC,DOC (day-1)")
    KN0: list = Field([0.005, 0.075, 0.2], description="minimum decay rate of RPON,LPON,DON (day-1)")
    KP0: list = Field([0.005, 0.075, 0.2], description="minimum decay rate of RPOP,LPOP,DOP (day-1)")
    KCalg: list = Field([0.0, 0.0, 0.0], description="algae effect on RPOC,LPOC,DOC decay (day-1.m3.g[C]-1)")
    KNalg: list = Field([0.0, 0.0, 0.0], description="algae effect on RPON,LPON,DON decay (day-1.m3.g[C]-1)")
    KPalg: list = Field([0.0, 0.0, 0.0], description="algae effect on RPOP,LPOP,DOP decay (day-1.m3.g[C]-1)")
    TRM: list = Field([20.0, 20.0, 20.0], description="reference temp. for (RPOM,LPOM,DOM) decay (oC)")
    KTRM: list = Field([0.069, 0.069, 0.069], description="temp. dependence for (RPOM,LPOM,DOM) decay (oC-1)")
    KSR0: list = Field([0.001, 0.001, 0.001], description="decay rates of SRPOC,SRPON,SRPOP (day-1)")
    TRSR: list = Field([20.0, 20.0, 20.0], description="reference temp. for (SRPOC,SRPON,SRPOP) decay (oC)")
    KTRSR: list = Field([0.069, 0.069, 0.069], description="temp. dependence for (SRPOC,SRPON,SRPOP) decay (oC-1)")
    KPIP: float = Field(0.0, description="dissolution rate of PIP (day-1)")
    KCD: float = Field(1.0, description="oxidation rate of COD at TRCOD (day-1)")
    TRCOD: float = Field(20.0, description="reference temp. for COD oxidation (oC)")
    KTRCOD: float = Field(0.041, description="temp. dependence for COD oxidation (oC-1)")
    KhCOD: float = Field(1.5, description="COD half saturation for COD oxidation (mg[O2]/L)")
    KhN: list = Field([0.01, 0.01, 0.01], description="nitrogen half saturation (mg/L)")
    KhP: list = Field([0.001, 0.001, 0.001], description="phosphorus half saturation (mg/L)")
    KhSal: list = Field(['1e6', '1e6', 0.1], description="salinity when PB growth is halved (PSU); (1e6: no salinity stress)")
    c2chl: list = Field([0.059, 0.059, 0.059], description="carbon to chlorophyll ratio (g[C]/mg[Chl])")
    n2c: list = Field([0.167, 0.167, 0.167], description="nitrogen to carbon ratio for phytoplankton")
    p2c: list = Field([0.02, 0.02, 0.02], description="phosphorus to carbon ratio for phytoplankton")
    o2c: float = Field(2.67, description="oxygen to carbon ratio in respiration")
    o2n: float = Field(4.33, description="oxygen to ammonium ratio (g[O2]/g[NH4])")
    dn2c: float = Field(0.933, description="mass of NO3 consumed per mass of DOC oxidized in denit. (g[N]/g[C])")
    an2c: float = Field(0.5, description="ratio of denit. rate to oxic DOC respiration rate")
    KhDO: list = Field([0.5, 0.5, 0.5], description="DO half saturation for PB's DOC excretion (mg/L)")
    KPO4p: float = Field(0.0, description="coefficient relating PO4 sorption to TSS")
    WRea: float = Field(0.0, description="baseline wind-induced reaeration coefficient for DO (day-1)")
    PBmin: list = Field([0.01, 0.01, 0.01], description="minimum PB concentration (mg[C]/L)")
    dz_flux: list = Field([1.0, 1.0], description="surface/bottom thikness (m) within which sflux/bflux are redistributed")

class SFM(NamelistBaseModel):
    btemp0: float = Field(5.0, description="initial temp. (oC)")
    bstc0: float = Field(0.1, description="surface transfer coefficient")
    bSTR0: float = Field(0.0, description="benthic stress (day)")
    bThp0: float = Field(0.0, description="consective days of hypoxia (day)")
    bTox0: float = Field(0.0, description="consective days of oxic condition after hypoxia event (day)")
    bNH40: float = Field(4.0, description="NH4")
    bNO30: float = Field(1.0, description="NO3")
    bPO40: float = Field(5.0, description="PO4")
    bH2S0: float = Field(250.0, description="H2S")
    bCH40: float = Field(40.0, description="CH4")
    bPOS0: float = Field(500.0, description="POS")
    bSA0: float = Field(500.0, description="SA")
    bPOC0: list = Field([1000.0, 3000.0, 5000.0], description="POC(G=1:3)")
    bPON0: list = Field([150.0, 500.0, 1500.0], description="PON(G=1:3)")
    bPOP0: list = Field([30.0, 300.0, 500.0], description="POP(G=1:3)")
    bdz: float = Field(0.1, description="sediment thickness (m)")
    bVb: str = Field('1.37e-5', description="burial rate (m.day-1); 1 cm/yr=2.74e-5 m.day-1")
    bsolid: list = Field([0.5, 0.5], description="sediment solid conc. in Layer 1 and Layer 2 (Kg.L-1)")
    bdiff: str = Field('1.8e-7', description="diffusion coefficient for sediment temp. (m2/s)")
    bTR: int = Field(20, description="reference temp. for sediment processes")
    bVpmin: str = Field('3.0e-6', description="minimum particle mixing velocity coefficient (m.day-1)")
    bVp: str = Field('1.2e-4', description="particle mixing velocity coefficient (m.day-1)")
    bVd: str = Field('1.0e-3', description="diffusion velocity coefficient (m.day-1)")
    bKTVp: float = Field(1.117, description="temp. dependece of particle mixing velocity")
    bKTVd: float = Field(1.08, description="temp. dependece of diffusion velocity")
    bKST: float = Field(0.03, description="1st order decay rate of benthic stress  (day-1)")
    bSTmax: float = Field(20.0, description="maximum value of benthic stress (day) (note: smaller than 1/bKST)")
    bKhDO_Vp: float = Field(4.0, description="DO half-saturation of particle mixing (mg/L)")
    bDOc_ST: float = Field(1.0, description="DO criteria for benthic stress (mg/L)")
    banoxic: float = Field(10.0, description="consective days of hypoxia causing maximum benthic stress (day)")
    boxic: float = Field(45.0, description="time lag for bethos recovery from hypoxia event (day)")
    bp2d: float = Field(0.0, description="ratio from mixing coef. to diffusion coef. (benthos enhanced effect)")
    bKC: list = Field([0.035, 0.0018, 0.0], description="decay rate of POC (3G class) at bTR (day-1)")
    bKN: list = Field([0.035, 0.0018, 0.0], description="decay rate of PON (3G class) at bTR (day-1)")
    bKP: list = Field([0.035, 0.0018, 0.0], description="decay rate of POP (3G class) at bTR (day-1)")
    bKTC: list = Field([1.1, 1.15, 1.17], description="temp. dependence of POC decay (oC-1)")
    bKTN: list = Field([1.1, 1.15, 1.17], description="temp. dependence of PON decay (oC-1)")
    bKTP: list = Field([1.1, 1.15, 1.17], description="temp. dependence of POP decay (oC-1)")
    bFCP: list = Field([0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01], description="Phyto POC into sed POC (G3,PB=1:3)")
    bFNP: list = Field([0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01], description="Phyto PON into sed PON (G3,PB=1:3)")
    bFPP: list = Field([0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01], description="Phyto POP into sed POP (G3,PB=1:3)")
    bFCM: list = Field([0.0, 0.43, 0.57], description="refractory POC into sed POC(3G)")
    bFNM: list = Field([0.0, 0.54, 0.46], description="refractory PON into sed PON(3G)")
    bFPM: list = Field([0.0, 0.43, 0.57], description="refractory POP into sed POP(3G)")
    bKNH4f: float = Field(0.2, description="NH4 reaction rate in freshwater  at bTR (1st layer) (m/day)")
    bKNH4s: float = Field(0.14, description="NH4 reaction rate in salty water at bTR (1st layer) (m/day)")
    bKTNH4: float = Field(1.08, description="temp. dependency for NH4 reaction (oC-1)")
    bKhNH4: float = Field(1.5, description="half-stauration NH4 for nitrification (g/m3)")
    bKhDO_NH4: float = Field(2.0, description="half-stauration DO for nitrification (g/m3)")
    bpieNH4: float = Field(1.0, description="partition coefficients of NH4 in Layer 1 & 2 (Kg-1.L)")
    bsaltn: float = Field(1.0, description="salinity criteria of fresh/salty water for NH4/NO3 reaction (PSU)")
    bKNO3f: float = Field(0.3, description="NO3 reaction rate in freshwater at bTR (1st layer) (m/day)")
    bKNO3s: float = Field(0.125, description="NO3 reaction rate in salty water at bTR (1st layer) (m/day)")
    bKNO3: float = Field(0.25, description="NO3 reaction rate (2nd layer) (m/day)")
    bKTNO3: float = Field(1.08, description="temp. dependency for NO3 reaction (oC-1)")
    bKH2Sd: float = Field(0.2, description="dissolved H2S reaction rate at bTR (1st layer) (m/day)")
    bKH2Sp: float = Field(0.4, description="particulate H2S reaction rate at bTR (1st layer) (m/day)")
    bKTH2S: float = Field(1.08, description="temp. dependency for H2S reaction (oC-1)")
    bpieH2Ss: float = Field(100.0, description="partition coefficient of NH4 in Layer 1 (Kg-1.L)")
    bpieH2Sb: float = Field(100.0, description="partition coefficient of NH4 in Layer 2 (Kg-1.L)")
    bKhDO_H2S: float = Field(8.0, description="O2 constant to normalize H2S oxidation (g[O2]/m3)")
    bsaltc: float = Field(1.0, description="salinity criteria of fresh/salty water for carbon reaction (PSU)")
    bKCH4: float = Field(0.2, description="CH4 reaction rate at bTR (1st layer) (m/day)")
    bKTCH4: float = Field(1.08, description="temp. dependency for CH4 reaction")
    bKhDO_CH4: float = Field(0.2, description="half-saturation DO for CH4 oxidation (g[O2]/m3)")
    bo2n: float = Field(2.86, description="oxygen to nitrogen ratio in sediment (denitrification)")
    bpiePO4: float = Field(50.0, description="partition coefficient of PO4 in Layer 2 (Kg-1.L)")
    bKOPO4f: float = Field(3000.0, description="oxygen dependency for PO4 sorption in freshwater in Layer 1")
    bKOPO4s: float = Field(300.0, description="oxygen dependency for PO4 sorption in salty water in Layer 1")
    bDOc_PO4: float = Field(1.0, description="DO criteria for PO4 sorptiona (g[O2]/m3)")
    bsaltp: float = Field(1.0, description="salinity criteria of fresh/salty water for PO4 partition (PSU)")
    bKS: float = Field(0.5, description="decay rate of POS (3G class) at bTR")
    bKTS: float = Field(1.1, description="temp. dependence of POS decay")
    bSIsat: float = Field(40.0, description="silica saturation conc. in pore water (g[Si]/m3)")
    bpieSI: float = Field(100.0, description="partition coefficient of silica in Layer 2 (Kg-1.L)")
    bKOSI: float = Field(10.0, description="oxygen dependency for silica sorption in Layer 1")
    bKhPOS: str = Field('5.0e4', description="POS half saturation for POS dissolution (g/m3)")
    bDOc_SI: float = Field(1.0, description="DO criteria for silica sorptiona (g[O2]/m3)")
    bJPOSa: float = Field(0.0, description="additional POS flux associated with POM detrius beside algea (g.m-2.day-1)")
    bFCs: list = Field([0.65, 0.255, 0.095], description="SAV POC into 3G sed 3G POC")
    bFNs: list = Field([0.65, 0.3, 0.05], description="SAV PON into 3G sed 3G PON")
    bFPs: list = Field([0.65, 0.255, 0.095], description="SAV POP into 3G sed 3G POP")
    bFCv: list = Field([0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095], description="VEG POC into sed POC (G3,PB=1:3)")
    bFNv: list = Field([0.65, 0.3, 0.05, 0.65, 0.3, 0.05, 0.65, 0.3, 0.05], description="VEG PON into sed PON (G3,PB=1:3)")
    bFPv: list = Field([0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095], description="VEG POP into sed POP (G3,PB=1:3)")

class SILICA(NamelistBaseModel):
    FSP: list = Field([0.9, 0.1], description="fractions of diatom silica into (SU,SA)")
    FSM: list = Field([0.5, 0.5], description="fractions of diatom metabolism Si into (SU,SA)")
    KS: float = Field(0.03, description="dissolution rate of SU at TRS (day-1)")
    TRS: float = Field(20.0, description="reference temp. for SU dissolution (oC)")
    KTRS: float = Field(0.092, description="temp. dependence for SU dissolution (oC-1)")
    KhS: list = Field([0.05, 0.0, 0.0], description="silica half saturation (mg/L); (0.0: no Si limitation)")
    s2c: list = Field([0.5, 0.0, 0.0], description="silica to carbon ratio for phytolankton; (0.0: no Si uptake)")
    KSAp: float = Field(0.0, description="coefficient relating Silicate(SA) sorption to TSS")

class ZB(NamelistBaseModel):
    zGPM: list = Field([0.0, 0.0, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], description="ZB predation rate(day-1); dim(prey=1:8, ZB=1:2)")
    zKhG: list = Field([0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175], description="reference prey conc.(mg/L); dim(prey=1:8, ZB=1:2)")
    zTGP: list = Field([25.0, 25.0], description="optimal temp. for ZB growth (oC)")
    zKTGP: list = Field([0.0035, 0.008, 0.025, 0.03], description="temp. dependence for ZB growth (T<=zTGP & T>zTGP); dim(ZB=1:2,1:2) (oC-2)")
    zAG: float = Field(0.75, description="ZB assimilation efficiency ratio (0-1)")
    zRG: float = Field(0.1, description="ZB respiration ratio when it grazes (0-1)")
    zMRT: list = Field([0.02, 0.02], description="ZB mortality rates (day-1)")
    zMTB: list = Field([0.254, 0.186], description="ZB metabolism rates (day-1)")
    zTMT: list = Field([20.0, 20.0], description="reference temp. for ZB metabolism (oC)")
    zKTMT: list = Field([0.0693, 0.0693], description="temp. dependence for ZB metabolism (oC-1)")
    zFCP: list = Field([0.35, 0.55, 0.1], description="fractions of ZB carbon into (RPOC,LPOC,DOC)")
    zFNP: list = Field([0.35, 0.5, 0.1, 0.05], description="fractions of ZB nitrogen into (RPON,LPON,DON,NH4)")
    zFPP: list = Field([0.1, 0.2, 0.5, 0.2], description="fractions of ZB Phosphorus into (RPOP,LPOP,DOP,PO4)")
    zFSP: list = Field([0.7, 0.25], description="fractions of ZB silica into (SU,SA)")
    zFCM: list = Field([0.1, 0.0], description="fractions of ZB metabolism carbon into DOC; dim(ZB=1:2)")
    zFNM: list = Field([0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1], description="fractions of ZB metabolism N. into (RPON,LPON,DON,NH4); dim(ZB=1:2,4)")
    zFPM: list = Field([0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1], description="fractions of ZB metabolism P. into (RPOP,LPOP,DOP,PO4); dim(ZB=1:2,4)")
    zFSM: list = Field([0.5, 0.4, 0.5, 0.6], description="fractions of ZB metabolism Si into (SU,SA); dim(ZB=1:2,2)")
    zKhDO: list = Field([0.5, 0.5], description="DO half saturation for ZB's DOC excretion (mg/L)")
    zn2c: list = Field([0.2, 0.2], description="nitrogen to carbon ratio for zooplankton")
    zp2c: list = Field([0.02, 0.02], description="phosphorus to carbon ratio for zooplankton")
    zs2c: list = Field([0.5, 0.5], description="silica to carbon ratio for zooplankton")
    z2pr: list = Field([0.5, 0.5], description="ratio converting ZB+PB biomass to predation rates on ZB (L.mg-1.day-1)")
    p2pr: float = Field(0.25, description="ratio converting ZB+PB biomass to predation rates on PB (L.mg-1.day-1)")

class PH_ICM(NamelistBaseModel):
    ppatch0: int = Field(-999, description="region flag for pH (1: ON all elem.; -999: spatial)")
    pKCACO3: float = Field(60.0, description="dissulution bewteen CaCO3 and Ca++")
    pKCA: float = Field(60.0, description="sediment surface transfer coefficient from CaCO3 to Ca++")
    pRea: float = Field(1.0, description="reaeration rate for CO2")
    inu_ph: int = Field(0, description="nudge option for pH model")

class SAV(NamelistBaseModel):
    spatch0: int = Field(-999, description="region flag for SAV. (1: ON all elem.; -999: spatial)")
    stleaf0: int = Field(-999, description="init conc of total sav leaf")
    ststem0: int = Field(-999, description="init conc of total sav stem")
    stroot0: int = Field(-999, description="init conc of total sav root")
    sGPM: float = Field(0.1, description="maximum growth rate (day-1)")
    sTGP: int = Field(32, description="optimal growth temperature (oC)")
    sKTGP: list = Field([0.003, 0.005], description="temp. dependence for growth (T<=sTGP & T>sTGP)")
    sFAM: float = Field(0.2, description="fraction of leaf production to active metabolism")
    sFCP: list = Field([0.6, 0.3, 0.1], description="fractions of production to leaf/stem/root biomass")
    sMTB: list = Field([0.02, 0.02, 0.02], description="metabolism rates of leaf/stem/root")
    sTMT: list = Field([20, 20, 20], description="reference temp. for leaf/stem/root metabolism")
    sKTMT: list = Field([0.069, 0.069, 0.069], description="temp. dependence of leaf/stem/root metabolism")
    sFCM: list = Field([0.05, 0.15, 0.3, 0.5], description="fractions of metabolism C into (RPOC,LPOC,DOC,CO2)")
    sFNM: list = Field([0.05, 0.15, 0.3, 0.5], description="fractions of metabolism N into (RPON,LPON,DON,NH4)")
    sFPM: list = Field([0.05, 0.1, 0.35, 0.5], description="fractions of metabolism P into (RPOP,LPOP,DOP,PO4)")
    sKhNw: float = Field(0.01, description="nitrogen half saturation in water column")
    sKhNs: float = Field(0.1, description="nitrogen half saturation in sediments")
    sKhNH4: float = Field(0.1, description="ammonium half saturation")
    sKhPw: float = Field(0.001, description="phosphorus half saturation in water column")
    sKhPs: float = Field(0.01, description="phosphorus half saturation in sediments")
    salpha: float = Field(0.006, description="init. slope of P-I curve (g[C]*m2/g[Chl]/E)")
    sKe: float = Field(0.045, description="light attenuation due to sav absorption")
    shtm: list = Field([0.054, 2.0], description="minimum (base) and  maximium canopy height")
    s2ht: list = Field([0.0036, 0.0036, 0.0], description="coeffs. converting (leaf,stem,root) to canopy height")
    sc2dw: float = Field(0.38, description="carbon to dry weight ratio of SAV")
    s2den: int = Field(10, description="coeff. computing SAV density from leaf")

class STEM(NamelistBaseModel):
    sn2c: float = Field(0.09, description="nitrogen to carbon ratio of sav")
    sp2c: float = Field(0.01, description="phosphorus to carbon ratio")
    so2c: float = Field(2.67, description="oxygen to carbon ratio")

class VEG(NamelistBaseModel):
    vpatch0: int = Field(-999, description="region flag for VEG. (1: ON all elem.; -999: spatial)")
    vtleaf0: list = Field([100.0, 100.0, 100.0], description="init conc to total veg leaf")
    vtstem0: list = Field([100.0, 100.0, 100.0], description="init conc to total veg stem")
    vtroot0: list = Field([30.0, 30.0, 30.0], description="init conc to total veg root")
    vGPM: list = Field([0.1, 0.1, 0.1], description="maximum growth rate (day-1)")
    vFAM: list = Field([0.2, 0.2, 0.2], description="fractions of leaf production to active metabolism")
    vTGP: list = Field([32.0, 32.0, 32.0], description="optimal growth temperature (oC)")
    vKTGP: list = Field([0.003, 0.003, 0.003, 0.005, 0.005, 0.005], description="temp. dependence(3,2) for growth (T<=vTGP & T>vTGP)")
    vFCP: list = Field([0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1], description="fractions of production to leaf/stem/root biomass; dim(veg,leaf/stem/root)")
    vMTB: list = Field([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01], description="metabolism rates of leaf/stem/root; dim(veg,leaf/stem/root)")
    vTMT: list = Field([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], description="reference temp. for leaf/stem/root metabolism")
    vKTMT: list = Field([0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069], description="temp. depdenpence for leaf/stem/root metabolism")
    vFNM: list = Field([0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5], description="fractions(3,4) of metabolism N into (RPON,LPON,DON,NH4)")
    vFPM: list = Field([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5], description="fractions(3,4) of metabolism P into (RPOP,LPOP,DOP,PO4)")
    vFCM: list = Field([0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5], description="fractions(3,4) of metabolism N into (RPOC,LPOC,DOC,CO2)")
    ivNc: int = Field(1, description="recycled veg N goes to (0: sediment; 1: water)")
    ivPc: int = Field(1, description="recycled veg P goes to (0: sediment; 1: water)")
    vKhNs: list = Field([0.1, 0.1, 0.1], description="nitrogen half saturation in sediments")
    vKhPs: list = Field([0.01, 0.01, 0.01], description="phosphorus half saturation in sediments")
    vScr: list = Field([35.0, 35.0, 35.0], description="reference sality for computing veg growth")
    vSopt: list = Field([35.0, 15.0, 0.0], description="optimal salinity for veg growth")
    vInun: list = Field([1.0, 1.0, 1.0], description="reference value for inundation stress (nondimensional)")
    ivNs: int = Field(1, description="N limitation on veg growth(0: OFF; 1: ON)")
    ivPs: int = Field(1, description="P limitation on veg growth(0: OFF; 1: ON)")
    ivMRT: int = Field(0, description="veg mortality term (0: OFF;  1: ON)")
    vTMR: list = Field([17.0, 17.0, 17.0, 17.0, 17.0, 17.0], description="reference temp(3,2) for leaf/stem mortality")
    vKTMR: list = Field([4.0, 4.0, 4.0, 4.0, 4.0, 4.0], description="temp dependence(3,2) for leaf/stem mortality")
    vMR0: list = Field([12.8, 12.8, 12.8, 12.8, 12.8, 12.8], description="base value(3,2) of temp effect on mortality (unit: None)")
    vMRcr: list = Field([15.0, 15.0, 15.0, 15.0, 15.0, 15.0], description="reference value(3,2) for computing mortality (unit: None)")
    valpha: list = Field([0.006, 0.006, 0.006], description="init. slope of P-I curve")
    vKe: list = Field([0.045, 0.045, 0.045], description="light attenuation from veg absorption")
    vht0: list = Field([0.054, 0.054, 0.054], description="base veg canopy hgt (vht)")
    vcrit: list = Field([250.0, 250.0, 250.0], description="critical mass for computing vht")
    v2ht: list = Field([0.0036, 0.0036, 0.0036, 0.001, 0.001, 0.001], description="coefs. convert mass(3,2) to canopy height")
    vc2dw: list = Field([0.38, 0.38, 0.38], description="carbon to dry weight ratio of VEG")
    v2den: list = Field([10, 10, 10], description="coeff. computing veg density")
    vp2c: list = Field([0.01, 0.01, 0.01], description="phosphorus to carbon ratio")
    vn2c: list = Field([0.09, 0.09, 0.09], description="nitrogen to carbon ratio")
    vo2c: list = Field([2.67, 2.67, 2.67], description="oxygen to carbon ratio")

class BAG(NamelistBaseModel):
    gpatch0: int = Field(-999, description="region flag for BA. (1: ON all elem.; -999: spatial)")
    BA0: float = Field(5.0, description="initial BA concentration (g[C].m-2)")
    gGPM: float = Field(2.25, description="BA maximum growth rate (day-1)")
    gTGP: float = Field(20.0, description="optimal temp. for BA growth (oC)")
    gKTGP: list = Field([0.004, 0.006], description="temp. dependence for BA growth (oC-2)")
    gMTB: float = Field(0.05, description="respiration rate at temp. of gTR (day-1)")
    gPRR: float = Field(0.1, description="predation rate at temp. of gTR (day-1)")
    gTR: float = Field(20.0, description="reference temperature for BA respiration (oC)")
    gKTR: float = Field(0.069, description="temp. dependence for BA respiration (oC-1)")
    galpha: float = Field(0.1, description="init. slope of P-I curve (m2/E)")
    gKSED: float = Field(0.0, description="light attenuation due to sediment (None)")
    gKBA: float = Field(0.01, description="light attenuation coef. due to BA self-shading (g[C]-1.m2)")
    gKhN: float = Field(0.01, description="nitrogen half saturation for BA growth (g[N]/m2)")
    gKhP: float = Field(0.001, description="phosphorus half saturation for BA growth (g[P]/2)")
    gp2c: float = Field(0.0167, description="phosphorus to carbon ratio")
    gn2c: float = Field(0.167, description="nitrogen to carbon ratio")
    go2c: float = Field(2.67, description="oxygen to carbon ratio")
    gFCP: list = Field([0.5, 0.45, 0.05], description="fraction of predation BA C into 3G classes in sediment")
    gFNP: list = Field([0.5, 0.45, 0.05], description="fraction of predation BA N into 3G classes in sediment")
    gFPP: list = Field([0.5, 0.45, 0.05], description="fraction of predation BA P into 3G classes in sediment")

class ERO(NamelistBaseModel):
    ierosion: int = Field(0, description="1: H2S flux; 2: POC flux; 3:  H2S")

class POC(NamelistBaseModel):
    erosion: int = Field(864, description="erosion rate (kg.m-2.day)")
    etau: str = Field('1.e-6', description="critical bottom shear stress (Pa)")
    eporo: float = Field(0.8, description="coefficinet in erosion formula (see code in icm_sfm.F90)")
    efrac: float = Field(0.5, description="fraction coefficinet in erosion formula (see code in icm_sfm.F90)")
    ediso: float = Field(2.5, description="H2S erosion coeffcient (see code in icm_sfm.F90)")
    dfrac: list = Field([0.02, 0.02], description="deposition fraction of POC (negative value: dfrac will be computed)")
    dWS_POC: list = Field([3.0, 3.0], description="coefficient in POC erosion (see code in icm_sfm.F90)")

class ICM(NamelistBaseModel):
    """
        
    This file was auto generated from a schism namelist file on 2023-11-20.
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
    MARCO: MARCO = MARCO()
    CORE: CORE = CORE()
    SFM: SFM = SFM()
    SILICA: SILICA = SILICA()
    ZB: ZB = ZB()
    PH_ICM: PH_ICM = PH_ICM()
    SAV: SAV = SAV()
    STEM: STEM = STEM()
    VEG: VEG = VEG()
    BAG: BAG = BAG()
    ERO: ERO = ERO()
    POC: POC = POC()
