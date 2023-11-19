from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel

class MICE_IN(NamelistBaseModel):
    ice_tests: int = Field(0)
    ihot_mice: int = Field(1)
    ice_advection: int = Field(6)
    ice_therm_on: int = Field(1)
    ievp: int = Field(2)
    ice_cutoff: float = Field(0.001)
    evp_rheol_steps: int = Field(500)
    mevp_rheol_steps: int = Field(500)
    delta_min: str = Field('1.0e-11')
    theta_io: float = Field(0.0)
    mevp_alpha1: float = Field(200.0)
    mevp_alpha2: float = Field(200.0)
    pstar: float = Field(27500.0)
    ellipse: float = Field(2.0)
    c_pressure: float = Field(20.0)
    niter_fct: int = Field(3)
    ice_gamma_fct: float = Field(0.25)
    h_ml0: float = Field(0.1)
    salt_ice: float = Field(5.0)
    salt_water: float = Field(34.0)

class MICE(NamelistBaseModel):
    """
        !parameter inputs via namelist convention.
    !(1)Use '' for chars; (2) integer values are fine for real vars/arrays;
    !(3) if multiple entries for a parameter are found, the last one wins - please avoid this
    !(4) array inputs follow column major and can spill to multiple lines
    !(5) space allowed before/after '='
    !(6) Not all required variables need to be present, but all that are present must belong to the list below. Best to list _all_ parameters.
    
    
    """
    MICE_IN: MICE_IN = MICE_IN()
