# This file was auto generated from a schism namelist file on 2024-08-28.

from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel


class MARCO(NamelistBaseModel):
    idelay: int = Field(0, description="")
    ndelay: int = Field(7, description="")
    ibgraze: int = Field(0, description="")
    idapt: int = Field(0, description="")
    alpha_corr: float = Field(1.25, description="")
    zeptic: float = Field(10.0, description="")
    iz2graze: int = Field(1, description="")
    iout_cosine: int = Field(0, description="")
    nspool_cosine: int = Field(30, description="")
    ico2s: int = Field(0, description="")
    ispm: int = Field(0, description="")
    spm0: float = Field(20.0, description="")
    ised: int = Field(1, description="")


class CORE(NamelistBaseModel):
    gmaxs: list = Field([2.0, 2.5], description="maximum growth rate")
    gammas: list = Field([0.2, 0.075], description="mortality rate")
    pis: list = Field([1.5, 1.5], description="ammonium inhibition")
    kno3s: list = Field([1.0, 3.0], description="NO3 half saturation")
    knh4s: list = Field([0.15, 0.45], description="NH4 half saturation")
    kpo4s: list = Field([0.1, 0.1], description="PO4 half saturation")
    kco2s: list = Field([50.0, 50.0], description="CO2 half saturation")
    ksio4: float = Field(4.5, description="SiO4 half saturation for diatom")
    kns: list = Field([0.0, 0.0], description="nighttime uptake rate of NH4")
    alphas: list = Field([0.1, 0.1], description="initial slopes of P-I curve")
    betas: list = Field([0.0, 0.0], description="slope for photo-inhibition")
    aks: list = Field(
        [0.75, 0.03, 0.066],
        description="light extinction coefficients: rKe=ak1+ak2*(S1+S2)+ak3*SPM",
    )
    betaz: list = Field([1.35, 0.4], description="maximum grazing rate")
    alphaz: list = Field([0.75, 0.75], description="assimilation rate")
    gammaz: list = Field([0.2, 0.2], description="mortality rate")
    kez: list = Field([0.2, 0.2], description="excretion rate")
    kgz: list = Field(
        [0.5, 0.25], description="reference prey concentration for grazing"
    )
    rhoz: list = Field(
        [0.6, 0.3, 0.1], description="prey preference factors of Z2 on (S2,Z1,DN)"
    )
    ipo4: int = Field(
        1, description="add additional PO4 from biogenic silica dissolution"
    )
    TR: float = Field(
        20.0,
        description="Reference temperature for temperature adjust for CoSiNE sink and source",
    )
    kox: float = Field(30.0, description="reference oxygen concentration for oxidation")
    wss2: float = Field(0.2, description="settling velocity of S2")
    wsdn: float = Field(1.0, description="settling velocity of DN")
    wsdsi: float = Field(1.0, description="settling velocity of DSi")
    si2n: float = Field(1.2, description="silica to nitrogen conversion coefficient")
    p2n: float = Field(
        0.0625, description="phosphorus to nitrogen conversion coefficient (1/16)"
    )
    o2no: float = Field(
        8.625, description="oxygen to nitrogen (NO3) conversion coefficient (138/16)"
    )
    o2nh: float = Field(
        6.625, description="oxygen to nitrogen (NH4) conversion coefficient (106/16)"
    )
    c2n: float = Field(7.3, description="carbon to nitrogen conversion coefficient")
    gamman: float = Field(0.07, description="nitrification coefficent")
    pco2a: float = Field(391.63, description="atmospheric CO2 concentration")
    kmdn: list = Field(
        [0.009, 0.075],
        description="remineralization coefficients for DN: rate=kmdn(1)*T+kmdn(2)",
    )
    kmdsi: list = Field(
        [0.0114, 0.015],
        description="remineralization coefficients for DSi: rate=kmdsi(1)*T+kmdsi(2)",
    )


class MISC(NamelistBaseModel):
    iws: int = Field(0, description="")
    NO3c: float = Field(2.0, description="mmol/m3")
    ws1: float = Field(2.5, description="")
    ws2: float = Field(2.0, description="")
    iclam: int = Field(0, description="")
    deltaZ: int = Field(1, description="meter")
    kcex: float = Field(0.002, description="day-1")
    Nperclam: float = Field(0.39032, description="mmol[N]")
    Wclam: str = Field("5.45e-3", description="clam weigh (g)")
    Fclam: int = Field(40, description="L.g[AFDW]-1.day-1, filtration rate")
    nclam0: int = Field(2000, description="")
    fS2: list = Field([0.1, 0.1, 0.8], description="")
    rkS2: list = Field(["4e-3", "1.0e-4", 0.0], description="time delay of 63 day")
    mkS2: list = Field([0.1, 0.01, 0.0], description="")
    fDN: list = Field([0.15, 0.1, 0.75], description="")
    rkDN: list = Field(["4e-3", "1.0e-4", 0.0], description="time delay of 63 day")
    mkDN: list = Field([0.1, 0.01, 0.0], description="")
    fDSi: list = Field([0.3, 0.3, 0.4], description="")
    rkDSi: list = Field(
        [0.004, "1e-4", 0.0], description="time delay of about half a month"
    )
    mkDSi: list = Field([0.1, 0.01, 0.0], description="")


class COSINE(NamelistBaseModel):
    """

    The full contents of the namelist file are shown below providing
    associated documentation for the objects:

    !parameter inputs via namelist convention.
    !(1) Use ' ' (single quotes) for chars;
    !(2) integer values are fine for real vars/arrays;
    !(3) if multiple entries for a parameter are found, the last one wins - please avoid this
    !(4) array inputs follow column major (like FORTRAN) and can spill to multiple lines
    !(5) space allowed before/after '='

    &MARCO
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !switches and marco parameters
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    !-----------------------------------------------------------------------
    !idelay: a 7-day delay for zooplankton predation
    !-----------------------------------------------------------------------
    idelay = 0
    ndelay = 7

    !-----------------------------------------------------------------------
    !ibgraze: bottom grazing function
    !-----------------------------------------------------------------------
    ibgraze = 0

    !-----------------------------------------------------------------------
    !idapt: light adaptation
    !-----------------------------------------------------------------------
    idapt = 0
    alpha_corr= 1.25
    zeptic= 10.0

    !-----------------------------------------------------------------------
    !iz2graze=0 : shut down Z2 grazing on S2, Z1, and DN
    !-----------------------------------------------------------------------
    iz2graze = 1

    !-----------------------------------------------------------------------
    !CoSiNE model station output option (need cstation.in with *.bp format)
    ! iout_cosine=0: turn off this option
    ! iout_cosine=1: all available diagnostic variables
    ! iout_cosine=2: state variables
    ! iout_cosine=3: state variables + source/sink terms
    ! iout_cosine=4: state variables + misc diagnostics
    ! iout_cosine=5: state variables + surface/bottom fluxes
    ! nspool_cosine: output interval (number of time step)
    !-----------------------------------------------------------------------
    iout_cosine=0
    nspool_cosine=30

    !-----------------------------------------------------------------------
    !ico2s=0: no CO2 limitation on phytoplankton growth
    !-----------------------------------------------------------------------
    ico2s = 0

    !-----------------------------------------------------------------------
    !ispm=0: constant Suspended Particlate Matter spm0 is used for while domain
    !ispm=1: spatial varying SPM from SPM.gr3 is used
    !ispm=2: use SED model to calculate SPM
    !-----------------------------------------------------------------------
    ispm = 0
    spm0 = 20.0

    !-----------------------------------------------------------------------
    !ised=1 : sediment flux model
    !-----------------------------------------------------------------------
    ised = 1
    /

    &CORE
    !------------------------------------------------------------------------
    !phytoplankton
    !------------------------------------------------------------------------
    gmaxs  = 2.0   2.5    !maximum growth rate
    gammas = 0.2   0.075  !mortality rate
    pis    = 1.5   1.5    !ammonium inhibition
    kno3s  = 1.0   3.0    !NO3 half saturation
    knh4s  = 0.15  0.45   !NH4 half saturation
    kpo4s  = 0.1   0.1    !PO4 half saturation
    kco2s  = 50.0  50.0   !CO2 half saturation
    ksio4  = 4.5          !SiO4 half saturation for diatom
    kns    = 0.0   0.0    !nighttime uptake rate of NH4
    alphas = 0.1   0.1    !initial slopes of P-I curve
    betas  = 0.0   0.0    !slope for photo-inhibition
    aks    = 0.75  0.03  0.066  !light extinction coefficients: rKe=ak1+ak2*(S1+S2)+ak3*SPM

    !------------------------------------------------------------------------
    !zooplankton
    !------------------------------------------------------------------------
    betaz  = 1.35  0.4    !maximum grazing rate
    alphaz = 0.75  0.75   !assimilation rate
    gammaz = 0.2   0.2    !mortality rate
    kez    = 0.2   0.2    !excretion rate
    kgz    = 0.5   0.25   !reference prey concentration for grazing
    rhoz   = 0.6   0.3   0.1  !prey preference factors of Z2 on (S2,Z1,DN)

    !------------------------------------------------------------------------
    !other
    !------------------------------------------------------------------------
    ipo4  = 1      !add additional PO4 from biogenic silica dissolution
    TR    = 20.0   !Reference temperature for temperature adjust for CoSiNE sink and source
    kox   = 30.0   !reference oxygen concentration for oxidation
    wss2  = 0.2    !settling velocity of S2
    wsdn  = 1.0    !settling velocity of DN
    wsdsi = 1.0    !settling velocity of DSi
    si2n  = 1.2    !silica to nitrogen conversion coefficient
    p2n   = 0.0625 !phosphorus to nitrogen conversion coefficient (1/16)
    o2no  = 8.625  !oxygen to nitrogen (NO3) conversion coefficient (138/16)
    o2nh  = 6.625  !oxygen to nitrogen (NH4) conversion coefficient (106/16)
    c2n   = 7.3    !carbon to nitrogen conversion coefficient
    gamman= 0.07   !nitrification coefficent
    pco2a = 391.63 !atmospheric CO2 concentration
    kmdn  = 0.009   0.075  !remineralization coefficients for DN: rate=kmdn(1)*T+kmdn(2)
    kmdsi = 0.0114  0.015  !remineralization coefficients for DSi: rate=kmdsi(1)*T+kmdsi(2)
    /

    &MISC
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !diatom sinking velocity depends on NO3
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    iws=0
    NO3c=2.0 !mmol/m3
    ws1=2.5
    ws2=2.0

    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !clam grazing model
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    iclam=0
    deltaZ= 1 !meter
    kcex=0.002  !day-1
    Nperclam=0.39032 !mmol[N]
    Wclam=5.45e-3  !clam weigh (g)
    Fclam=40  !L.g[AFDW]-1.day-1, filtration rate
    nclam0=2000

    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !sediment model
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    !-----------------------------------------------------------------------
    !parameters related to S2 in sediment
    !fS2:  partitioning coefficient from S2 in water column into sediment S2
    !rkS2: changing rate of remineralization rate for sediment S2
    !mkS2: maximum remineralization rate for sediment S2
    !-----------------------------------------------------------------------
    fS2=  0.1    0.1     0.8
    rkS2= 4e-3   1.0e-4  0.0   !time delay of 63 day
    mkS2= 0.1    0.01    0.0

    !-----------------------------------------------------------------------
    !parameters related to DN in sediment
    !fdDN: partitioning coefficient from DN in water column into sediment DN
    !rkDN: changing rate of remineralization rate for sediment DN
    !mkDN: maximum remineralization rate for sediment DN
    !-----------------------------------------------------------------------
    fDN=  0.15   0.10    0.75
    rkDN= 4e-3   1.0e-4  0.0 !time delay of 63 day
    mkDN= 0.1    0.01    0.0

    !-----------------------------------------------------------------------
    !parameters related to DSi in sediment
    !fDSi:  partitioning coefficient from DSi in water column into sediment DSi
    !rkDSi: changing rate of remineralization rate for sediment DSi
    !mkDSi: maximum remineralization rate for sediment DSi
    !-----------------------------------------------------------------------
    fDSi=  0.3   0.3   0.4
    rkDSi= 0.004 1e-4  0.0 !time delay of about half a month
    mkDSi= 0.1   0.01  0.0
    /

    """

    marco: MARCO | None = Field(default=None)
    core: CORE | None = Field(default=None)
    misc: MISC | None = Field(default=None)
