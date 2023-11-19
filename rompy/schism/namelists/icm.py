from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel

class MARCO(NamelistBaseModel):
    nsub: int = Field(1)
    iKe: int = Field(0)
    Ke0: float = Field(0.26)
    KeC: float = Field(0.017)
    KeS: float = Field(0.07)
    KeSalt: float = Field(-0.02)
    tss2c: float = Field(6.0)
    iLight: int = Field(0)
    alpha: list = Field([8.0, 8.0, 8.0])
    iPR: int = Field(1)
    PRR: list = Field([0.1, 0.2, 0.05])
    wqc0: list = Field([1.0, 0.5, 0.05, 1.0, 0.5, 0.5, 0.15, 0.15, 0.05, 0.01, 0.05, 0.005, 0.005, 0.01, 0.05, 0.0, 12.0])
    WSP: list = Field([0.3, 0.1, 0.0, 0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0])
    WSPn: list = Field([0.3, 0.1, 0.0, 0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0])
    iSilica: int = Field(0)
    iZB: int = Field(0)
    iPh: int = Field(0)
    iCBP: int = Field(0)
    isav_icm: int = Field(0)
    iveg_icm: int = Field(0)
    iSed: int = Field(1)
    iBA: int = Field(0)
    iRad: int = Field(0)
    isflux: int = Field(0)
    ibflux: int = Field(0)
    iout_icm: int = Field(0)
    nspool_icm: int = Field(24)
    iLimit: int = Field(0)
    idry_icm: int = Field(0)

class CORE(NamelistBaseModel):
    GPM: list = Field([2.5, 2.8, 3.5])
    TGP: list = Field([15.0, 22.0, 27.0])
    KTGP: list = Field([0.005, 0.004, 0.003, 0.008, 0.006, 0.004])
    MTR: list = Field([0.0, 0.0, 0.0])
    MTB: list = Field([0.01, 0.02, 0.03])
    TMT: list = Field([20.0, 20.0, 20.0])
    KTMT: list = Field([0.0322, 0.0322, 0.0322])
    FCP: list = Field([0.35, 0.3, 0.2, 0.55, 0.5, 0.5, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    FNP: list = Field([0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0])
    FPP: list = Field([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0])
    FCM: list = Field([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
    FNM: list = Field([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    FPM: list = Field([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Nit: float = Field(0.07)
    TNit: float = Field(27.0)
    KTNit: list = Field([0.0045, 0.0045])
    KhDOn: float = Field(1.0)
    KhDOox: float = Field(0.5)
    KhNO3dn: float = Field(0.1)
    KC0: list = Field([0.005, 0.075, 0.2])
    KN0: list = Field([0.005, 0.075, 0.2])
    KP0: list = Field([0.005, 0.075, 0.2])
    KCalg: list = Field([0.0, 0.0, 0.0])
    KNalg: list = Field([0.0, 0.0, 0.0])
    KPalg: list = Field([0.0, 0.0, 0.0])
    TRM: list = Field([20.0, 20.0, 20.0])
    KTRM: list = Field([0.069, 0.069, 0.069])
    KSR0: list = Field([0.001, 0.001, 0.001])
    TRSR: list = Field([20.0, 20.0, 20.0])
    KTRSR: list = Field([0.069, 0.069, 0.069])
    KPIP: float = Field(0.0)
    KCD: float = Field(1.0)
    TRCOD: float = Field(20.0)
    KTRCOD: float = Field(0.041)
    KhCOD: float = Field(1.5)
    KhN: list = Field([0.01, 0.01, 0.01])
    KhP: list = Field([0.001, 0.001, 0.001])
    KhSal: list = Field(['1e6', '1e6', 0.1])
    c2chl: list = Field([0.059, 0.059, 0.059])
    n2c: list = Field([0.167, 0.167, 0.167])
    p2c: list = Field([0.02, 0.02, 0.02])
    o2c: float = Field(2.67)
    o2n: float = Field(4.33)
    dn2c: float = Field(0.933)
    an2c: float = Field(0.5)
    KhDO: list = Field([0.5, 0.5, 0.5])
    KPO4p: float = Field(0.0)
    WRea: float = Field(0.0)
    PBmin: list = Field([0.01, 0.01, 0.01])
    dz_flux: list = Field([1.0, 1.0])

class SFM(NamelistBaseModel):
    btemp0: float = Field(5.0)
    bstc0: float = Field(0.1)
    bSTR0: float = Field(0.0)
    bThp0: float = Field(0.0)
    bTox0: float = Field(0.0)
    bNH40: float = Field(4.0)
    bNO30: float = Field(1.0)
    bPO40: float = Field(5.0)
    bH2S0: float = Field(250.0)
    bCH40: float = Field(40.0)
    bPOS0: float = Field(500.0)
    bSA0: float = Field(500.0)
    bPOC0: list = Field([1000.0, 3000.0, 5000.0])
    bPON0: list = Field([150.0, 500.0, 1500.0])
    bPOP0: list = Field([30.0, 300.0, 500.0])
    bdz: float = Field(0.1)
    bVb: str = Field('1.37e-5')
    bsolid: list = Field([0.5, 0.5])
    bdiff: str = Field('1.8e-7')
    bTR: int = Field(20)
    bVpmin: str = Field('3.0e-6')
    bVp: str = Field('1.2e-4')
    bVd: str = Field('1.0e-3')
    bKTVp: float = Field(1.117)
    bKTVd: float = Field(1.08)
    bKST: float = Field(0.03)
    bSTmax: float = Field(20.0)
    bKhDO_Vp: float = Field(4.0)
    bDOc_ST: float = Field(1.0)
    banoxic: float = Field(10.0)
    boxic: float = Field(45.0)
    bp2d: float = Field(0.0)
    bKC: list = Field([0.035, 0.0018, 0.0])
    bKN: list = Field([0.035, 0.0018, 0.0])
    bKP: list = Field([0.035, 0.0018, 0.0])
    bKTC: list = Field([1.1, 1.15, 1.17])
    bKTN: list = Field([1.1, 1.15, 1.17])
    bKTP: list = Field([1.1, 1.15, 1.17])
    bFCP: list = Field([0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01])
    bFNP: list = Field([0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01])
    bFPP: list = Field([0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01])
    bFCM: list = Field([0.0, 0.43, 0.57])
    bFNM: list = Field([0.0, 0.54, 0.46])
    bFPM: list = Field([0.0, 0.43, 0.57])
    bKNH4f: float = Field(0.2)
    bKNH4s: float = Field(0.14)
    bKTNH4: float = Field(1.08)
    bKhNH4: float = Field(1.5)
    bKhDO_NH4: float = Field(2.0)
    bpieNH4: float = Field(1.0)
    bsaltn: float = Field(1.0)
    bKNO3f: float = Field(0.3)
    bKNO3s: float = Field(0.125)
    bKNO3: float = Field(0.25)
    bKTNO3: float = Field(1.08)
    bKH2Sd: float = Field(0.2)
    bKH2Sp: float = Field(0.4)
    bKTH2S: float = Field(1.08)
    bpieH2Ss: float = Field(100.0)
    bpieH2Sb: float = Field(100.0)
    bKhDO_H2S: float = Field(8.0)
    bsaltc: float = Field(1.0)
    bKCH4: float = Field(0.2)
    bKTCH4: float = Field(1.08)
    bKhDO_CH4: float = Field(0.2)
    bo2n: float = Field(2.86)
    bpiePO4: float = Field(50.0)
    bKOPO4f: float = Field(3000.0)
    bKOPO4s: float = Field(300.0)
    bDOc_PO4: float = Field(1.0)
    bsaltp: float = Field(1.0)
    bKS: float = Field(0.5)
    bKTS: float = Field(1.1)
    bSIsat: float = Field(40.0)
    bpieSI: float = Field(100.0)
    bKOSI: float = Field(10.0)
    bKhPOS: str = Field('5.0e4')
    bDOc_SI: float = Field(1.0)
    bJPOSa: float = Field(0.0)
    bFCs: list = Field([0.65, 0.255, 0.095])
    bFNs: list = Field([0.65, 0.3, 0.05])
    bFPs: list = Field([0.65, 0.255, 0.095])
    bFCv: list = Field([0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095])
    bFNv: list = Field([0.65, 0.3, 0.05, 0.65, 0.3, 0.05, 0.65, 0.3, 0.05])
    bFPv: list = Field([0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095])

class SILICA(NamelistBaseModel):
    FSP: list = Field([0.9, 0.1])
    FSM: list = Field([0.5, 0.5])
    KS: float = Field(0.03)
    TRS: float = Field(20.0)
    KTRS: float = Field(0.092)
    KhS: list = Field([0.05, 0.0, 0.0])
    s2c: list = Field([0.5, 0.0, 0.0])
    KSAp: float = Field(0.0)

class ZB(NamelistBaseModel):
    zGPM: list = Field([0.0, 0.0, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    zKhG: list = Field([0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175])
    zTGP: list = Field([25.0, 25.0])
    zKTGP: list = Field([0.0035, 0.008, 0.025, 0.03])
    zAG: float = Field(0.75)
    zRG: float = Field(0.1)
    zMRT: list = Field([0.02, 0.02])
    zMTB: list = Field([0.254, 0.186])
    zTMT: list = Field([20.0, 20.0])
    zKTMT: list = Field([0.0693, 0.0693])
    zFCP: list = Field([0.35, 0.55, 0.1])
    zFNP: list = Field([0.35, 0.5, 0.1, 0.05])
    zFPP: list = Field([0.1, 0.2, 0.5, 0.2])
    zFSP: list = Field([0.7, 0.25])
    zFCM: list = Field([0.1, 0.0])
    zFNM: list = Field([0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1])
    zFPM: list = Field([0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1])
    zFSM: list = Field([0.5, 0.4, 0.5, 0.6])
    zKhDO: list = Field([0.5, 0.5])
    zn2c: list = Field([0.2, 0.2])
    zp2c: list = Field([0.02, 0.02])
    zs2c: list = Field([0.5, 0.5])
    z2pr: list = Field([0.5, 0.5])
    p2pr: float = Field(0.25)

class PH_ICM(NamelistBaseModel):
    ppatch0: int = Field(-999)
    pKCACO3: float = Field(60.0)
    pKCA: float = Field(60.0)
    pRea: float = Field(1.0)
    inu_ph: int = Field(0)

class SAV(NamelistBaseModel):
    spatch0: int = Field(-999)
    stleaf0: int = Field(-999)
    ststem0: int = Field(-999)
    stroot0: int = Field(-999)
    sGPM: float = Field(0.1)
    sTGP: int = Field(32)
    sKTGP: list = Field([0.003, 0.005])
    sFAM: float = Field(0.2)
    sFCP: list = Field([0.6, 0.3, 0.1])
    sMTB: list = Field([0.02, 0.02, 0.02])
    sTMT: list = Field([20, 20, 20])
    sKTMT: list = Field([0.069, 0.069, 0.069])
    sFCM: list = Field([0.05, 0.15, 0.3, 0.5])
    sFNM: list = Field([0.05, 0.15, 0.3, 0.5])
    sFPM: list = Field([0.05, 0.1, 0.35, 0.5])
    sKhNw: float = Field(0.01)
    sKhNs: float = Field(0.1)
    sKhNH4: float = Field(0.1)
    sKhPw: float = Field(0.001)
    sKhPs: float = Field(0.01)
    salpha: float = Field(0.006)
    sKe: float = Field(0.045)
    shtm: list = Field([0.054, 2.0])
    s2ht: list = Field([0.0036, 0.0036, 0.0])
    sc2dw: float = Field(0.38)
    s2den: int = Field(10)

class STEM(NamelistBaseModel):
    sn2c: float = Field(0.09)
    sp2c: float = Field(0.01)
    so2c: float = Field(2.67)

class VEG(NamelistBaseModel):
    vpatch0: int = Field(-999)
    vtleaf0: list = Field([100.0, 100.0, 100.0])
    vtstem0: list = Field([100.0, 100.0, 100.0])
    vtroot0: list = Field([30.0, 30.0, 30.0])
    vGPM: list = Field([0.1, 0.1, 0.1])
    vFAM: list = Field([0.2, 0.2, 0.2])
    vTGP: list = Field([32.0, 32.0, 32.0])
    vKTGP: list = Field([0.003, 0.003, 0.003, 0.005, 0.005, 0.005])
    vFCP: list = Field([0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
    vMTB: list = Field([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01])
    vTMT: list = Field([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    vKTMT: list = Field([0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069])
    vFNM: list = Field([0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5])
    vFPM: list = Field([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5])
    vFCM: list = Field([0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5])
    ivNc: int = Field(1)
    ivPc: int = Field(1)
    vKhNs: list = Field([0.1, 0.1, 0.1])
    vKhPs: list = Field([0.01, 0.01, 0.01])
    vScr: list = Field([35.0, 35.0, 35.0])
    vSopt: list = Field([35.0, 15.0, 0.0])
    vInun: list = Field([1.0, 1.0, 1.0])
    ivNs: int = Field(1)
    ivPs: int = Field(1)
    ivMRT: int = Field(0)
    vTMR: list = Field([17.0, 17.0, 17.0, 17.0, 17.0, 17.0])
    vKTMR: list = Field([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    vMR0: list = Field([12.8, 12.8, 12.8, 12.8, 12.8, 12.8])
    vMRcr: list = Field([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
    valpha: list = Field([0.006, 0.006, 0.006])
    vKe: list = Field([0.045, 0.045, 0.045])
    vht0: list = Field([0.054, 0.054, 0.054])
    vcrit: list = Field([250.0, 250.0, 250.0])
    v2ht: list = Field([0.0036, 0.0036, 0.0036, 0.001, 0.001, 0.001])
    vc2dw: list = Field([0.38, 0.38, 0.38])
    v2den: list = Field([10, 10, 10])
    vp2c: list = Field([0.01, 0.01, 0.01])
    vn2c: list = Field([0.09, 0.09, 0.09])
    vo2c: list = Field([2.67, 2.67, 2.67])

class BAG(NamelistBaseModel):
    gpatch0: int = Field(-999)
    BA0: float = Field(5.0)
    gGPM: float = Field(2.25)
    gTGP: float = Field(20.0)
    gKTGP: list = Field([0.004, 0.006])
    gMTB: float = Field(0.05)
    gPRR: float = Field(0.1)
    gTR: float = Field(20.0)
    gKTR: float = Field(0.069)
    galpha: float = Field(0.1)
    gKSED: float = Field(0.0)
    gKBA: float = Field(0.01)
    gKhN: float = Field(0.01)
    gKhP: float = Field(0.001)
    gp2c: float = Field(0.0167)
    gn2c: float = Field(0.167)
    go2c: float = Field(2.67)
    gFCP: list = Field([0.5, 0.45, 0.05])
    gFNP: list = Field([0.5, 0.45, 0.05])
    gFPP: list = Field([0.5, 0.45, 0.05])

class ERO(NamelistBaseModel):
    ierosion: int = Field(0)

class POC(NamelistBaseModel):
    erosion: int = Field(864)
    etau: str = Field('1.e-6')
    eporo: float = Field(0.8)
    efrac: float = Field(0.5)
    ediso: float = Field(2.5)
    dfrac: list = Field([0.02, 0.02])
    dWS_POC: list = Field([3.0, 3.0])

class ICM(NamelistBaseModel):
    """
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
