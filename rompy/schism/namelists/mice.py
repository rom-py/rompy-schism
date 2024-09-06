# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import Optional

from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Mice_in(NamelistBaseModel):
    ice_tests: Optional[int] = Field(None, description="box test flag")
    ihot_mice: Optional[int] = Field(
        None, description="0: cold start 1: restart 2: hotstart_from_HYCOM"
    )
    ice_advection: Optional[int] = Field(
        None,
        description="advection on/off 3: upwind 4: center-difference 5: tvd 6: tvd-up 7: TVD_Casulli",
    )
    ice_therm_on: Optional[int] = Field(
        None, description="ice thermodynamics on/off flag"
    )
    ievp: Optional[int] = Field(None, description="1: EVP; 2: mEVP")
    ice_cutoff: Optional[float] = Field(
        None,
        description="cut-off thickness [m] or fraction for ice. No ice velocity if *<=ice_cuttoff",
    )
    evp_rheol_steps: Optional[int] = Field(
        None, description="the number of sybcycling steps in EVP"
    )
    mevp_rheol_steps: Optional[int] = Field(
        None, description="the number of iterations in mEVP"
    )
    delta_min: Optional[str] = Field(
        None, description="(1/s) Limit for minimum divergence (Hibler, Hunke"
    )
    theta_io: Optional[float] = Field(
        None, description="ice/ocean rotation angle. [degr]"
    )
    mevp_alpha1: Optional[float] = Field(
        None, description="const used in mEVP (constitutive eq)"
    )
    mevp_alpha2: Optional[float] = Field(
        None, description="const used in mEVP for momentum eq"
    )
    pstar: Optional[float] = Field(None, description="[N/m^2]")
    ellipse: Optional[float] = Field(None, description="ellipticity")
    c_pressure: Optional[float] = Field(None, description="C [-]")
    niter_fct: Optional[int] = Field(
        None, description="# of iterartions in higher-order solve"
    )
    ice_gamma_fct: Optional[float] = Field(None, description="smoothing parameter")
    h_ml0: Optional[float] = Field(None, description="ocean mixed layer depth [m]")
    salt_ice: Optional[float] = Field(None, description="salinity for ice [PSU] (>=0)")
    salt_water: Optional[float] = Field(
        None, description="salinity for water [PSU] (>=0)"
    )


class Mice(NamelistBaseModel):
    """

    The full contents of the namelist file are shown below providing
    associated documentation for the objects:

    !parameter inputs via namelist convention.
    !(1)Use '' for chars; (2) integer values are fine for real vars/arrays;
    !(3) if multiple entries for a parameter are found, the last one wins - please avoid this
    !(4) array inputs follow column major and can spill to multiple lines
    !(5) space allowed before/after '='
    !(6) Not all required variables need to be present, but all that are present must belong to the list below. Best to list _all_ parameters.

    &mice_in
      ice_tests = 0  !box test flag
      ihot_mice = 1  !0: cold start 1: restart 2: hotstart_from_HYCOM
      ice_advection = 6 !advection on/off 3: upwind 4: center-difference 5: tvd 6: tvd-up 7: TVD_Casulli
      ice_therm_on = 1 !ice thermodynamics on/off flag
      ievp=2 !1: EVP; 2: mEVP
      ice_cutoff=0.001 !cut-off thickness [m] or fraction for ice. No ice velocity if *<=ice_cuttoff
      evp_rheol_steps=500  ! the number of sybcycling steps in EVP
      mevp_rheol_steps=500  ! the number of iterations in mEVP
      delta_min=1.0e-11     ! (1/s) Limit for minimum divergence (Hibler, Hunke
                           ! normally use 2.0e-9, which does much stronger
                           ! limiting; valid for both VP and EVP
      theta_io=0.       ! ice/ocean rotation angle. [degr]
      mevp_alpha1=200. !const used in mEVP (constitutive eq)
      mevp_alpha2=200. !const used in mEVP for momentum eq
      pstar=27500. ![N/m^2]
      ellipse=2.  !ellipticity
      c_pressure=20.0  !C [-]
      !FCT
      niter_fct=3 !# of iterartions in higher-order solve
      ice_gamma_fct=0.25 ! smoothing parameter

      !Thermodynamics
      h_ml0=0.1 !ocean mixed layer depth [m]
      salt_ice=5. !salinity for ice [PSU] (>=0)
      salt_water=34. !salinity for water [PSU] (>=0)
    /

    """

    mice_in: Optional[Mice_in] = Field(default=None)
