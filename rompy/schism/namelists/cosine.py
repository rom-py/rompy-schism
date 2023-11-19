from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel

class MARCO(NamelistBaseModel):
    idelay: int = Field(0)
    ndelay: int = Field(7)
    ibgraze: int = Field(0)
    idapt: int = Field(0)
    alpha_corr: float = Field(1.25)
    zeptic: float = Field(10.0)
    iz2graze: int = Field(1)
    iout_cosine: int = Field(0)
    nspool_cosine: int = Field(30)
    ico2s: int = Field(0)
    ispm: int = Field(0)
    spm0: float = Field(20.0)
    ised: int = Field(1)

class CORE(NamelistBaseModel):
    gmaxs: list = Field([2.0, 2.5])
    gammas: list = Field([0.2, 0.075])
    pis: list = Field([1.5, 1.5])
    kno3s: list = Field([1.0, 3.0])
    knh4s: list = Field([0.15, 0.45])
    kpo4s: list = Field([0.1, 0.1])
    kco2s: list = Field([50.0, 50.0])
    ksio4: float = Field(4.5)
    kns: list = Field([0.0, 0.0])
    alphas: list = Field([0.1, 0.1])
    betas: list = Field([0.0, 0.0])
    aks: list = Field([0.75, 0.03, 0.066])
    betaz: list = Field([1.35, 0.4])
    alphaz: list = Field([0.75, 0.75])
    gammaz: list = Field([0.2, 0.2])
    kez: list = Field([0.2, 0.2])
    kgz: list = Field([0.5, 0.25])
    rhoz: list = Field([0.6, 0.3, 0.1])
    ipo4: int = Field(1)
    TR: float = Field(20.0)
    kox: float = Field(30.0)
    wss2: float = Field(0.2)
    wsdn: float = Field(1.0)
    wsdsi: float = Field(1.0)
    si2n: float = Field(1.2)
    p2n: float = Field(0.0625)
    o2no: float = Field(8.625)
    o2nh: float = Field(6.625)
    c2n: float = Field(7.3)
    gamman: float = Field(0.07)
    pco2a: float = Field(391.63)
    kmdn: list = Field([0.009, 0.075])
    kmdsi: list = Field([0.0114, 0.015])

class MISC(NamelistBaseModel):
    iws: int = Field(0)
    NO3c: float = Field(2.0)
    ws1: float = Field(2.5)
    ws2: float = Field(2.0)
    iclam: int = Field(0)
    deltaZ: int = Field(1)
    kcex: float = Field(0.002)
    Nperclam: float = Field(0.39032)
    Wclam: str = Field('5.45e-3')
    Fclam: int = Field(40)
    nclam0: int = Field(2000)
    fS2: list = Field([0.1, 0.1, 0.8])
    rkS2: list = Field(['4e-3', '1.0e-4', 0.0])
    mkS2: list = Field([0.1, 0.01, 0.0])
    fDN: list = Field([0.15, 0.1, 0.75])
    rkDN: list = Field(['4e-3', '1.0e-4', 0.0])
    mkDN: list = Field([0.1, 0.01, 0.0])
    fDSi: list = Field([0.3, 0.3, 0.4])
    rkDSi: list = Field([0.004, '1e-4', 0.0])
    mkDSi: list = Field([0.1, 0.01, 0.0])

class COSINE(NamelistBaseModel):
    """
        !parameter inputs via namelist convention.
    !(1) Use ' ' (single quotes) for chars;
    !(2) integer values are fine for real vars/arrays;
    !(3) if multiple entries for a parameter are found, the last one wins - please avoid this
    !(4) array inputs follow column major (like FORTRAN) and can spill to multiple lines
    !(5) space allowed before/after '='
    
    
    """
    MARCO: MARCO = MARCO()
    CORE: CORE = CORE()
    MISC: MISC = MISC()
