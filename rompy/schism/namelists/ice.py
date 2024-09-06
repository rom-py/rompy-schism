# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import Optional

from pydantic import Field
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Ice_in(NamelistBaseModel):
    ice_tests: Optional[int] = Field(None, description="box test flag")
    ice_advection: Optional[int] = Field(None, description="advection on/off")
    ice_therm_on: Optional[int] = Field(
        None, description="ice thermodynamics on/off flag"
    )
    ievp: Optional[int] = Field(None, description="1: EVP; 2: mEVP")
    ice_cutoff: Optional[str] = Field(
        None,
        description="cut-off thickness [m] or fraction for ice. No ice velocity if *<=ice_cuttoff",
    )
    evp_rheol_steps: Optional[int] = Field(
        None, description="the number of sybcycling steps in EVP"
    )
    mevp_rheol_steps: Optional[int] = Field(
        None, description="the number of iterations in mEVP"
    )
    ice_atmos_stress_form: Optional[int] = Field(None, description="")
    cdwin0: Optional[str] = Field(
        None, description="needed if ice_atmos_stress_form=0 (const Cdw)"
    )
    delta_min: Optional[str] = Field(
        None, description="(1/s) Limit for minimum divergence (Hibler, Hunke"
    )
    theta_io: Optional[float] = Field(
        None,
        description="ice/ocean rotation angle. [degr]. Usually 0 unless vgrid is too coarse",
    )
    mevp_coef: Optional[int] = Field(None, description="")
    mevp_alpha1: Optional[float] = Field(
        None, description="const used in mEVP (constitutive eq), if mevp_coef=0"
    )
    mevp_alpha2: Optional[float] = Field(
        None, description="const used in mEVP for momentum eq, if mevp_coef=0"
    )
    mevp_alpha3: Optional[float] = Field(None, description="if mevp_coef=1")
    mevp_alpha4: Optional[str] = Field(None, description="if mevp_coef=1")
    pstar: Optional[float] = Field(None, description="[N/m^2]")
    ellipse: Optional[float] = Field(None, description="ellipticity")
    c_pressure: Optional[float] = Field(None, description="C [-]")
    ncyc_fct: Optional[int] = Field(None, description="# of subcycling in transport")
    niter_fct: Optional[int] = Field(
        None, description="# of iterartions in higher-order solve"
    )
    ice_gamma_fct: Optional[float] = Field(
        None, description="smoothing parameter; 1 for max positivity preserving"
    )
    depth_ice_fct: Optional[float] = Field(
        None, description="cut off depth (m) for non-FCT"
    )
    h_ml0: Optional[float] = Field(None, description="ocean mixed layer depth [m]")
    salt_ice: Optional[float] = Field(None, description="salinity for ice [PSU] (>=0)")
    salt_water: Optional[float] = Field(
        None, description="salinity for water [PSU] (>=0)"
    )
    lead_closing: Optional[float] = Field(
        None,
        description="lead closing parameter [m] - larger values slow down freezing-up but increase sea ice thickness",
    )
    saterm: Optional[float] = Field(
        None, description="Semter const -smaller value could slow down melting"
    )
    albsn: Optional[float] = Field(None, description="Albedo: frozen snow")
    albsnm: Optional[float] = Field(None, description="melting snow (<=albsn)")
    albi: Optional[float] = Field(None, description="frozen ice (<=albsn)")
    albm: Optional[float] = Field(None, description="melting ice (<=albi)")


class Ice(NamelistBaseModel):
    """

    The full contents of the namelist file are shown below providing
    associated documentation for the objects:

    !parameter inputs via namelist convention.
    !(1)Use '' for chars; (2) integer values are fine for real vars/arrays;
    !(3) if multiple entries for a parameter are found, the last one wins - please avoid this
    !(4) array inputs follow column major and can spill to multiple lines
    !(5) space allowed before/after '='
    !(6) Not all required variables need to be present, but all that are present must belong to the list below. Best to list _all_ parameters.

    &ice_in
      !All values shown are default unless otherwise stated
      ice_tests = 0  !box test flag
      ice_advection = 1 !advection on/off
      ice_therm_on = 1 !ice thermodynamics on/off flag
      ievp=2 !1: EVP; 2: mEVP
      ice_cutoff=1.e-3 !cut-off thickness [m] or fraction for ice. No ice velocity if *<=ice_cuttoff
      evp_rheol_steps=200  ! the number of sybcycling steps in EVP
      mevp_rheol_steps=200  ! the number of iterations in mEVP
      !ice_atmos_stress_form: 0-const Cd; 1: FESOM formulation
      ice_atmos_stress_form=1
      cdwin0=2.e-3 !needed if ice_atmos_stress_form=0 (const Cdw)

      delta_min=2.0e-9     ! (1/s) Limit for minimum divergence (Hibler, Hunke
                           ! normally use 2.0e-9, which does much stronger
                           ! limiting; valid for both VP and EVP
      theta_io=0.       ! ice/ocean rotation angle. [degr]. Usually 0 unless vgrid is too coarse

      !Options for specifying 2 relax coefficients in mEVP only (mevp_coef)
      !0: constant (mevp_alpha[12] below); 1: both coefficients equal to: mevp_alpha3/tanh(mevp_alpha4*area/dt_ice)
      !In this case, mevp_alpha3 is the min, and at the finer end of mesh, the coeff's are approximately ~mevp_alpha4^-1
      mevp_coef=0
      mevp_alpha1=200. !const used in mEVP (constitutive eq), if mevp_coef=0
      mevp_alpha2=200. !const used in mEVP for momentum eq, if mevp_coef=0
      mevp_alpha3=200. !if mevp_coef=1
      mevp_alpha4=2.e-2 !if mevp_coef=1

      pstar=15000. ![N/m^2]
      ellipse=2.  !ellipticity
      c_pressure=20.0  !C [-]

      !FCT
      ncyc_fct=1 !# of subcycling in transport
      niter_fct=3 !# of iterartions in higher-order solve
      ice_gamma_fct=0.25 ! smoothing parameter; 1 for max positivity preserving
      !non-FCT zone is delineated by h<=depth_ice_fct OR depth=0 in ice_fct.gr3
      depth_ice_fct=5. !cut off depth (m) for non-FCT

      !Thermodynamics
      h_ml0=0.1 !ocean mixed layer depth [m]
      salt_ice=5. !salinity for ice [PSU] (>=0)
      salt_water=34. !salinity for water [PSU] (>=0)
      lead_closing=0.5 !lead closing parameter [m] - larger values slow down freezing-up but increase sea ice thickness
      Saterm=0.5 !Semter const -smaller value could slow down melting
      albsn=0.85 !Albedo: frozen snow
      albsnm=0.75  !melting snow (<=albsn)
      albi=0.75  !frozen ice (<=albsn)
      albm=0.66  !melting ice (<=albi)
    /

    """

    ice_in: Optional[Ice_in] = Field(default=None)
