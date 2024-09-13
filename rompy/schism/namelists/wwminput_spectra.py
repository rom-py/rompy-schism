# This file was auto generated from a SCHISM namelist file on 2024-09-13.

from datetime import datetime
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Hotfile(NamelistBaseModel):
    lhotf: Optional[bool] = Field(True, description="Write hotfile")
    begtc: Optional[str] = Field(
        "19980901.000000", description="Starting time of hotfile writing"
    )
    deltc: Optional[int] = Field(3600, description="time between hotfile writes")
    unitc: Optional[str] = Field("SEC", description="unit used above")
    endtc: Optional[str] = Field(
        "19980901.060000", description="Ending time of hotfile writing"
    )
    lcyclehot: Optional[bool] = Field(True, description="Applies only to netcdf")
    hotstyle_out: Optional[int] = Field(
        2, description="1: binary hotfile of data as output"
    )
    multipleout: Optional[int] = Field(
        0, description="0: hotfile in a single file (binary or netcdf)"
    )
    filehot_out: Optional[str] = Field(
        "hotfile_out_WWM.nc", description="name of hot outputs"
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "Model":
        if self.BEGTC >= self.ENDTC:
            raise ValueError("BEGTC must be earlier than ENDTC")
        return self

    @model_validator(mode="after")
    def validate_filehot_out_consistency(self) -> "Model":
        if self.HOTSTYLE_OUT == 2 and not self.FILEHOT_OUT.endswith(".nc"):
            raise ValueError("FILEHOT_OUT must end with .nc when HOTSTYLE_OUT is 2")
        return self


class Proc(NamelistBaseModel):
    procname: Optional[str] = Field("limon", description="Project Name")
    dimmode: Optional[int] = Field(
        2,
        description="Mode of run (ex: 1 = 1D, 2 = 2D) always 2D when coupled to SCHISM",
    )
    lstea: Optional[bool] = Field(False, description="steady mode; under development")
    lqstea: Optional[bool] = Field(
        False,
        description="Quasi-Steady Mode; In this case WWM-II is doing subiterations defined as DELTC/NQSITER unless QSCONVI is not reached",
    )
    lsphe: Optional[bool] = Field(False, description="Spherical coordinates (lon/lat)")
    lnautin: Optional[bool] = Field(
        True,
        description="Nautical convention for all inputs given in degrees (suggestion: T)",
    )
    lmono_in: Optional[bool] = Field(False, description="")
    lmono_out: Optional[bool] = Field(False, description="")
    lnautout: Optional[bool] = Field(
        True, description="Nautical output of all quantities in degrees"
    )
    begtc: Optional[str] = Field(
        "19980901.000000",
        description="Time for start the simulation, ex:yyyymmdd. hhmmss",
    )
    deltc: Optional[int] = Field(
        5, description="Time step (MUST match dt*nstep_wwm in SCHISM!)"
    )
    unitc: Optional[str] = Field("SEC", description="Unity of time step")
    endtc: Optional[str] = Field(
        "19980901.060000",
        description="Time for stop the simulation, ex:yyyymmdd. hhmmss",
    )
    dmin: Optional[float] = Field(
        0.01, description="Minimum water depth. This must be the same as h0 in SCHISM"
    )

    @model_validator(mode="after")
    def validate_time_range(self):
        start = datetime.strptime(self.BEGTC, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.ENDTC, "%Y%m%d.%H%M%S")
        if end <= start:
            raise ValueError("ENDTC must be later than BEGTC")
        return self


class Coupl(NamelistBaseModel):
    lcpl: Optional[bool] = Field(
        True,
        description="Couple with current model ... main switch - keep it on for SCHISM-WWM",
    )
    lroms: Optional[bool] = Field(False, description="ROMS (set as F)")
    ltimor: Optional[bool] = Field(False, description="TIMOR (set as F)")
    lshyfem: Optional[bool] = Field(False, description="SHYFEM (set as F)")
    radflag: Optional[str] = Field("LON", description="")
    letot: Optional[bool] = Field(
        False,
        description="Option to compute the wave induced radiation stress. If .T. the radiation stress is based on the integrated wave spectrum",
    )
    nlvt: Optional[int] = Field(
        10, description="Number of vertical Layers; not used with SCHISM"
    )
    dtcoup: Optional[float] = Field(
        600.0, description="Couple time step - not used when coupled to SCHISM"
    )


class Grid(NamelistBaseModel):
    lcird: Optional[bool] = Field(True, description="Full circle in directional space")
    lstag: Optional[bool] = Field(
        False,
        description="Stagger directional bins with a half Dtheta; may use T only for regular grid to avoid char. line aligning with grid line",
    )
    mindir: Optional[float] = Field(
        0.0,
        description="Minimum direction for simulation (unit: degrees; nautical convention; 0: from N; 90: from E); not used if LCIRD = .T.",
    )
    maxdir: Optional[float] = Field(
        360.0,
        description="Maximum direction for simulation (unit: degrees); may be < MINDIR; not used if LCIRD = .T.",
    )
    mdc: Optional[int] = Field(36, description="Number of directional bins")
    frlow: Optional[float] = Field(
        0.04,
        description="Low frequency limit of the discrete wave period (Hz; 1/period)",
    )
    frhigh: Optional[float] = Field(
        1.0, description="High frequency limit of the discrete wave period."
    )
    msc: Optional[int] = Field(36, description="Number of frequency bins")
    igridtype: Optional[int] = Field(
        3,
        description="Gridtype used. 1 ~ XFN, 2 ~ WWM-PERIODIC, 3 ~ SCHISM, 4 ~ OLD WWM GRID",
    )
    filegrid: Optional[str] = Field(
        "hgrid_WWM.gr3",
        description="Name of the grid file. hgridi_WWM.gr3 if IGRIDTYPE = 3 (SCHISM)",
    )
    lslop: Optional[bool] = Field(False, description="Bottom Slope limiter (default=F)")
    slmax: Optional[float] = Field(0.2, description="Max Slope;")
    lvar1d: Optional[bool] = Field(
        False, description="For 1d-mode if variable dx is used; not used with SCHISM"
    )

    @model_validator(mode="after")
    def validate_mindir_maxdir(self) -> "Model":
        if not self.LCIRD and self.MINDIR >= self.MAXDIR:
            raise ValueError("MINDIR must be less than MAXDIR when LCIRD is False")
        return self

    @model_validator(mode="after")
    def validate_frequency_range(self) -> "Model":
        if self.FRLOW >= self.FRHIGH:
            raise ValueError("FRLOW must be less than FRHIGH")
        return self

    @model_validator(mode="after")
    def validate_filegrid_igridtype(self) -> "Model":
        if self.IGRIDTYPE == 3 and self.FILEGRID != "hgrid_WWM.gr3":
            raise ValueError("FILEGRID must be hgrid_WWM.gr3 when IGRIDTYPE is 3")
        return self

    @model_validator(mode="after")
    def validate_lvar1d_igridtype(self) -> "Model":
        if self.LVAR1D and self.IGRIDTYPE == 3:
            raise ValueError("LVAR1D is not used with SCHISM (IGRIDTYPE=3)")
        return self


class Bouc(NamelistBaseModel):
    lbcse: Optional[bool] = Field(
        False, description="The wave boundary data is time dependent"
    )
    lbinter: Optional[bool] = Field(
        False,
        description="Do interpolation in time if LBCSE=T (not available for quasi-steady mode within the subtime steps)",
    )
    lbcwa: Optional[bool] = Field(True, description="Parametric Wave Spectra")
    linhom: Optional[bool] = Field(False, description="Non-uniform wave b.c. in space")
    lbcsp: Optional[bool] = Field(
        False,
        description="Specify (non-parametric) wave spectra, specified in 'FILEWAVE' below",
    )
    lindsprdeg: Optional[bool] = Field(
        False,
        description="If 1-d wave spectra are read this flag defines whether the input for the directional spreading is in degrees (true) or exponent (false)",
    )
    lparmdir: Optional[bool] = Field(
        False,
        description="If LPARMDIR is true than directional spreading is read from WBDS and must be in exponential format at this time, only valid for 1d Spectra",
    )
    filewave: Optional[str] = Field(
        "wwmbnd.gr3", description="Boundary file including discrete wave spectra"
    )
    lbsp1d: Optional[bool] = Field(
        False,
        description="1D (freq. space only) format for FILEWAVE if LBCSP=T and LINHOM=F",
    )
    lbsp2d: Optional[bool] = Field(
        False, description="not functional (freq. + directional space)"
    )
    begtc: Optional[str] = Field(
        "19980901.000000", description="Begin time of the wave boundary file (FILEWAVE)"
    )
    deltc: Optional[int] = Field(1, description="Time step in FILEWAVE")
    unitc: Optional[str] = Field("HR", description="Unit can be HR, MIN, SEC")
    endtc: Optional[str] = Field("19981002.000000", description="End time")
    filebound: Optional[str] = Field(
        "wwmbnd.gr3", description="Boundary file defining boundary and Neumann nodes."
    )
    iboundformat: Optional[int] = Field(
        1, description="1 ~ WWM, 3 ~ WW3 (2D spectra in netcdf format only - LBCWA=T)."
    )
    wbhs: Optional[float] = Field(
        4.0, description="Hs at the boundary for parametric spectra"
    )
    wbss: Optional[float] = Field(
        2.0,
        description="1 or -1: Pierson-Moskowitz, 2 or -2: JONSWAP, 3 or -3: all in one BIN,",
    )
    wbtp: Optional[float] = Field(
        8.0,
        description="Tp at the boundary (sec); mean or peak depending on the sign of WBSS",
    )
    wbdm: Optional[float] = Field(
        90.0, description="Avg. Wave Direction at the boundary"
    )
    wbdsms: Optional[float] = Field(
        1.0, description="Directional spreading value in degrees (1) or as exponent (2)"
    )
    wbds: Optional[float] = Field(
        20.0, description="Directional spreading at the boundary (degrees/exponent)"
    )
    wbgauss: Optional[float] = Field(
        0.1, description="factor for gaussian distribution if WBSS=1"
    )
    wbpken: Optional[float] = Field(
        3.3, description="Peak enhancement factor for Jonswap Spectra if WBSS=2"
    )
    ncdf_hs_name: Optional[str] = Field(
        "hs",
        description="NETCDF var. name for the significant wave height (normally it is just 'hs')",
    )
    ncdf_dir_name: Optional[str] = Field(
        "dir",
        description="NETCDF var. name for the mean wave direction (normally it is just 'dir')",
    )
    ncdf_spr_name: Optional[str] = Field(
        "spr",
        description="NETCDF var. name for the mean directional spreading (normally it is just 'spr')",
    )
    ncdf_fp_name: Optional[str] = Field(
        "fp",
        description="NETCDF var. name for the peak freq. (normally it is just 'fp')",
    )
    ncdf_f02_name: Optional[str] = Field(
        "t02",
        description="NETCDF var. name for the zero down crossing freq. (normally it is just 't02')",
    )

    @model_validator(mode="after")
    def validate_lbinter_lbcse(self):
        if self.LBINTER and not self.LBCSE:
            raise ValueError("LBINTER can only be True if LBCSE is True")
        return self

    @model_validator(mode="after")
    def validate_lbsp1d_conditions(self):
        if self.LBSP1D and not (self.LBCSP and not self.LINHOM):
            raise ValueError(
                "LBSP1D can only be True if LBCSP is True and LINHOM is False"
            )
        return self

    @model_validator(mode="after")
    def validate_time_range(self):
        begin = datetime.strptime(self.BEGTC, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.ENDTC, "%Y%m%d.%H%M%S")
        if end <= begin:
            raise ValueError("ENDTC must be later than BEGTC")
        return self


class Engs(NamelistBaseModel):
    mesnl: Optional[int] = Field(
        1,
        description="Nonlinear Interaction NL4 , 1 ~ on, 0 ~ off (Discrete Interaction approx.)",
    )
    mesin: Optional[int] = Field(
        1, description="Wind input: Ardhuin et al. (1) (use LSOURCESWAM = F);"
    )
    ifric: Optional[int] = Field(
        1,
        description="Formulation for atmospheric boundary layer, (IFRIC = 1 for MESIN = 1, IFRIC = 4 for MESIN=3);",
    )
    mesbf: Optional[int] = Field(
        1,
        description="Bottom friction: 1 - JONSWAP (Default); 2 - Madsen et al. (1989); 3 - SHOWEX",
    )
    fricc: Optional[float] = Field(
        0.067,
        description="if MESBF=1: JONSWAP bottom friction coefficient [0.038,0.067]. If MESBF=2: physical bottom roughness (ignored if given in rough.gr3). If MESBF=3: D50 (if negative read from SHOWEX_D50.gr3)",
    )
    mesbr: Optional[int] = Field(
        1, description="Shallow water wave breaking; 0: off; 1: on"
    )
    ibreak: Optional[int] = Field(
        1, description="Wave breaking formulation: 1 - Battjes and Janssen (1978)"
    )
    icrit: Optional[int] = Field(
        1,
        description="Wave breaking criterion: 1   - Constant breaker index (gamma) or gamma_TG defined with BRCR",
    )
    brcr: Optional[float] = Field(
        0.78,
        description="either gamma, default is 0.73 for IBREAK=1,5 or gamma_TG, default is 0.42 for IBREAK=2,3 or biphase_ref, default is -4pi/9 = -1.3963 for IBREAK=4",
    )
    a_brcr: Optional[float] = Field(0.76, description="cf ICRIT = 4, 5")
    b_brcr: Optional[float] = Field(0.29, description="cf ICRIT = 4, 5")
    min_brcr: Optional[float] = Field(0.25, description="cf ICRIT = 4, 5")
    max_brcr: Optional[float] = Field(0.8, description="cf ICRIT = 4, 5")
    a_biph: Optional[float] = Field(
        0.2, description="Biphase coefficient, default 0.2 (intended for IBREAK=3)"
    )
    br_coef_method: Optional[int] = Field(
        1, description="Method for the breaking coefficient: 1 - constant, 2 - adaptive"
    )
    b_alp: Optional[float] = Field(
        0.5, description="breaking coefficient. If BR_COEF_METHOD = 2, B_ALP ~ 40"
    )
    zprof_break: Optional[int] = Field(
        2,
        description="Vertical distribution function of wave breaking source term, only used in 3D run",
    )
    bc_break: Optional[int] = Field(
        1, description="Apply depth-limited breaking at the boundaries: 1 - On; 0 - Off"
    )
    iroller: Optional[int] = Field(
        0,
        description="Wave roller model (e.g., see Uchiyama et al., 2010): 1 - On; 0 - Off; not used at the moment",
    )
    alprol: Optional[float] = Field(
        0.85,
        description="Alpha coefficient for the wave roller model (between 0 and 1): 1 - full conversion; 0 - no energy transferred to the roller",
    )
    meveg: Optional[int] = Field(
        0, description="Vegetation on/off. If on, isav must = 1 in param.nml"
    )
    lmaxetot: Optional[bool] = Field(
        True,
        description="Limit shallow water wave height by wave breaking limiter (default=T)",
    )
    mesds: Optional[int] = Field(
        1,
        description="Formulation for the whitecapping source function; same value as MESIN",
    )
    mestr: Optional[int] = Field(
        1,
        description="Formulation for the triad 3 wave interactions (MESTR = 0 (off), MESTR = 1 (Lumped Triad Approx. (LTA)), MESTR = 2 (corrected version of LTA by Salmon et al. (2016)))",
    )
    trico: Optional[float] = Field(
        0.1, description="proportionality const. (\alpha_EB); default is 0.1"
    )
    trira: Optional[float] = Field(
        2.5,
        description="ratio of max. freq. considered in triads over mean freq.; 2.5 is suggested",
    )
    triurs: Optional[float] = Field(
        0.1,
        description="critical Ursell number; if Ursell # < TRIURS; triads are not computed",
    )

    @model_validator(mode="after")
    def validate_mesin_lsourceswam(self) -> "Model":
        if self.MESIN == 1 and self.LSOURCESWAM:
            raise ValueError("When MESIN=1, LSOURCESWAM should be False")
        if self.MESIN == 2 and not self.LSOURCESWAM:
            raise ValueError("When MESIN=2, LSOURCESWAM should be True")
        return self

    @model_validator(mode="after")
    def validate_ifric_mesin(self) -> "Model":
        if self.MESIN == 1 and self.IFRIC != 1:
            raise ValueError("When MESIN=1, IFRIC should be 1")
        if self.MESIN == 3 and self.IFRIC != 4:
            raise ValueError("When MESIN=3, IFRIC should be 4")
        return self

    @model_validator(mode="after")
    def validate_fricc_mesbf(self) -> "Model":
        if self.MESBF == 1 and not 0.038 <= self.FRICC <= 0.067:
            raise ValueError("When MESBF=1, FRICC should be between 0.038 and 0.067")
        return self

    @model_validator(mode="after")
    def validate_b_alp_br_coef_method(self) -> "Model":
        if self.BR_COEF_METHOD == 2 and self.B_ALP < 30:
            raise ValueError("When BR_COEF_METHOD=2, B_ALP should be around 40")
        return self

    @model_validator(mode="after")
    def validate_mesds_mesin(self) -> "Model":
        if self.MESDS != self.MESIN:
            raise ValueError("MESDS should have the same value as MESIN")
        return self


class Sin4(NamelistBaseModel):
    zwnd: Optional[str] = Field("", description="")
    alpha0: Optional[str] = Field("", description="")
    z0max: Optional[str] = Field("", description="")
    betamax: Optional[str] = Field("", description="")
    sinthp: Optional[str] = Field("", description="")
    zalp: Optional[str] = Field("", description="")
    tauwshelter: Optional[str] = Field("", description="")
    swellfpar: Optional[str] = Field("", description="")
    swellf: Optional[str] = Field("", description="")
    swellf2: Optional[str] = Field("-", description="")
    swellf3: Optional[str] = Field("", description="")
    swellf4: Optional[str] = Field("", description="")
    swellf5: Optional[str] = Field("", description="")
    swellf6: Optional[str] = Field("", description="")
    swellf7: Optional[str] = Field("", description="")
    z0rat: Optional[str] = Field("", description="")
    sinbr: Optional[float] = Field(0.0, description="")

    @model_validator(mode="after")
    def validate_swellf_limits(self):
        if self.SWELLF6 >= self.SWELLF7:
            raise ValueError("SWELLF6 must be less than SWELLF7")
        return self


class Sds4(NamelistBaseModel):
    sdsc1: Optional[str] = Field("", description="")
    fxpm3: Optional[str] = Field("", description="")
    fxfm3: Optional[str] = Field("", description="")
    fxfmage: Optional[str] = Field("", description="")
    sdsc2: Optional[str] = Field("-", description="")
    sdscum: Optional[str] = Field("-", description="")
    sdsstrain: Optional[str] = Field("", description="")
    sdsc4: Optional[str] = Field("", description="")
    sdsc5: Optional[str] = Field("", description="")
    sdsc6: Optional[str] = Field("", description="")
    sdsbr: Optional[str] = Field("", description="")
    sdsbr2: Optional[str] = Field("", description="")
    sdsp: Optional[str] = Field("", description="")
    sdsiso: Optional[str] = Field("", description="")
    sdsbck: Optional[str] = Field("", description="")
    sdsabk: Optional[str] = Field("", description="")
    sdspbk: Optional[str] = Field("", description="")
    sdsbint: Optional[str] = Field("", description="")
    sdshck: Optional[str] = Field("", description="")
    sdsdth: Optional[str] = Field("", description="")
    sdscos: Optional[str] = Field("", description="")
    sdsbrf1: Optional[str] = Field("", description="")
    sdsbrfdf: Optional[str] = Field("", description="")
    sdsbm0: Optional[str] = Field("", description="")
    sdsbm1: Optional[str] = Field("", description="")
    sdsbm2: Optional[str] = Field("", description="")
    sdsbm3: Optional[str] = Field("", description="")
    sdsbm4: Optional[str] = Field("", description="")
    sdshfgen: Optional[str] = Field("", description="")
    sdslfgen: Optional[str] = Field("", description="")
    whitecapwidth: Optional[str] = Field("", description="")
    fxincut: Optional[str] = Field("", description="")
    fxdscut: Optional[float] = Field(0.0, description="")


class Nums(NamelistBaseModel):
    icomp: Optional[int] = Field(3, description="")
    amethod: Optional[int] = Field(7, description="")
    aspar_local_level: Optional[str] = Field("", description="")
    smethod: Optional[int] = Field(1, description="")
    dmethod: Optional[int] = Field(2, description="")
    rtheta: Optional[float] = Field(
        0.5,
        description="Weighing factor for DMETHOD = 1, not really useful since Crank Nicholson integration can only be monotone for CFL .le. 2",
    )
    litersplit: Optional[bool] = Field(
        False,
        description="T: double Strang split; F: simple split (more efficient). Default: F",
    )
    lfilterth: Optional[bool] = Field(False, description="")
    maxcflth: Optional[float] = Field(
        1.0, description="Max Cfl in Theta space; used only if LFILTERTH=T"
    )
    fmethod: Optional[int] = Field(1, description="")
    lfiltersig: Optional[bool] = Field(
        False, description="Limit the advection velocity in freq. space (usually F)"
    )
    maxcflsig: Optional[float] = Field(
        1.0, description="Max Cfl in freq. space; used only if LFILTERSIG=T"
    )
    llimt: Optional[bool] = Field(
        True,
        description="Switch on/off Action limiter, Action limiter must mostly be turned on.",
    )
    lsigbound: Optional[bool] = Field(
        False, description="Theta space on wet land/island boundary"
    )
    lthbound: Optional[bool] = Field(
        False, description="Sigma space on wet land/island boundary"
    )
    lsoubound: Optional[bool] = Field(
        False,
        description="Source Terms on wet land/island boundary. Use T if SMETHOD=6",
    )
    melim: Optional[int] = Field(1, description="Formulation for the action limiter")
    limfak: Optional[float] = Field(
        0.1,
        description="Proportionality coefficient for the action limiter MAX_DAC_DT = Limfak * Limiter; see notes above for value",
    )
    ldifr: Optional[bool] = Field(
        False,
        description="Use phase decoupled diffraction approximation according to Holthuijsen et al. (2003) (usually T; if crash, use F)",
    )
    idiffr: Optional[int] = Field(
        1,
        description="Extended WAE accounting for higher order effects WAE becomes nonlinear; 1: Holthuijsen et al. ; 2: Liau et al. ; 3: Toledo et al. (in preparation)",
    )
    lconv: Optional[bool] = Field(
        False,
        description="Estimate convergence criterian and write disk (quasi-steady - qstea.out)",
    )
    lcfl: Optional[bool] = Field(
        False, description="Write out CFL numbers; use F to save time"
    )
    nqsiter: Optional[int] = Field(
        10,
        description="# of quasi-steady (Q-S) sub-divisions within each WWM time step (trial and errors)",
    )
    qsconv1: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill abs. wave height criteria EPSH1",
    )
    qsconv2: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill rel. wave height criteria EPSH2",
    )
    qsconv3: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill sum. rel. wave action criteria EPSH3",
    )
    qsconv4: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave period criteria EPSH4",
    )
    qsconv5: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave steepness criteria EPSH5",
    )
    lexpimp: Optional[bool] = Field(
        False,
        description="Use implicit schemes for freq. lower than given below by FREQEXP; used only if ICOMP=0",
    )
    freqexp: Optional[float] = Field(
        0.1,
        description="Minimum frequency for explicit schemes; only used if LEXPIMP=T and ICOMP=0",
    )
    epsh1: Optional[float] = Field(
        0.01,
        description="Convergence criteria for rel. wave height ! EPSH1 < CONVK1 = REAL(ABS(HSOLD(IP)-HS2)/HS2)",
    )
    epsh2: Optional[float] = Field(
        0.01,
        description="Convergence criteria for abs. wave height ! EPSH2 < CONVK2 = REAL(ABS(HS2-HSOLD(IP)))",
    )
    epsh3: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. sum of wave action ! EPSH3 < CONVK3 = REAL(ABS(SUMACOLD(IP)-SUMAC)/SUMAC)",
    )
    epsh4: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. avg. wave steepness criteria ! EPSH4 < CONVK4 = REAL(ABS(KHS2-KHSOLD(IP))/KHSOLD(IP))",
    )
    epsh5: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. avg. waveperiod ! EPSH5 < REAL(ABS(TM02-TM02OLD(IP))/TM02OLD(IP))",
    )
    lvector: Optional[bool] = Field(
        False,
        description="Use optmized propagation routines for large high performance computers e.g. at least more than 128 CPU. Try LVECTOR=F first.",
    )
    ivector: Optional[int] = Field(
        2, description="USed if LVECTOR=T; Different flavours of communications"
    )
    ladvtest: Optional[bool] = Field(
        False,
        description="for testing the advection schemes, testcase will be added soon",
    )
    lchkconv: Optional[bool] = Field(
        True,
        description="needs to set to .true. for quasi-steady mode. in order to compute the QSCONVi criteria and check them",
    )
    nb_block: Optional[str] = Field("", description="")
    wae_solverthr: Optional[str] = Field("", description="")
    maxiter: Optional[str] = Field("", description="")
    lsourceswam: Optional[bool] = Field(
        False, description="Use ECMWF WAM formualtion for deep water physics."
    )
    lnaninfchk: Optional[bool] = Field(False, description="")
    lzeta_setup: Optional[bool] = Field(False, description="")
    zeta_meth: Optional[str] = Field("", description="")
    pmin: Optional[str] = Field("", description="")
    block_gauss_seidel: Optional[bool] = Field(True, description="")
    lnonl: Optional[bool] = Field(False, description="")
    l_solver_norm: Optional[bool] = Field(False, description="")

    @model_validator(mode="after")
    def validate_amethod_icomp(self):
        if self.ICOMP == 0 and self.AMETHOD in [1, 2, 3]:
            return True
        elif self.ICOMP > 0 and self.AMETHOD in [1, 2, 3, 4, 5, 6, 7]:
            return True
        raise ValueError("Invalid AMETHOD and ICOMP combination")
        return self

    @model_validator(mode="after")
    def validate_maxcflth_lfilterth(self):
        if self.LFILTERTH and self.MAXCFLTH <= 0:
            raise ValueError("MAXCFLTH must be positive when LFILTERTH is True")
        return self

    @model_validator(mode="after")
    def validate_maxcflsig_lfiltersig(self):
        if self.LFILTERSIG and self.MAXCFLSIG <= 0:
            raise ValueError("MAXCFLSIG must be positive when LFILTERSIG is True")
        return self

    @model_validator(mode="after")
    def validate_lsoubound_smethod(self):
        if self.SMETHOD == 6 and not self.LSOUBOUND:
            raise ValueError("LSOUBOUND should be True when SMETHOD is 6")
        return self

    @model_validator(mode="after")
    def validate_lexpimp_icomp(self):
        if self.LEXPIMP and self.ICOMP != 0:
            raise ValueError("LEXPIMP should only be True when ICOMP is 0")
        return self

    @model_validator(mode="after")
    def validate_freqexp_lexpimp_icomp(self):
        if self.LEXPIMP and self.ICOMP == 0 and self.FREQEXP <= 0:
            raise ValueError(
                "FREQEXP must be positive when LEXPIMP is True and ICOMP is 0"
            )
        return self

    @model_validator(mode="after")
    def validate_ivector_lvector(self):
        if self.LVECTOR and self.IVECTOR not in range(1, 7):
            raise ValueError("When LVECTOR is True, IVECTOR must be between 1 and 6")
        return self


class History(NamelistBaseModel):
    begtc: Optional[str] = Field(
        "19980901.000000", description="Start output time, yyyymmdd. hhmmss;"
    )
    deltc: Optional[int] = Field(
        1,
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )
    unitc: Optional[str] = Field("SEC", description="Unit")
    endtc: Optional[str] = Field(
        "19980910.000000", description="Stop time output, yyyymmdd. hhmmss"
    )
    definetc: Optional[int] = Field(
        86400, description="Time for definition of history files"
    )
    outstyle: Optional[str] = Field(
        "NO", description="output option - use 'NO' for no output"
    )
    multipleout: Optional[int] = Field(0, description="0: output in a single netcdf")
    use_single_out: Optional[bool] = Field(
        True,
        description="T: Use single precision in the output of model variables (default)",
    )
    paramwrite: Optional[bool] = Field(
        True, description="T/F: Write the physical parametrization and"
    )
    gridwrite: Optional[bool] = Field(
        True, description="T/F: Write the grid in the netcdf history file (default T)"
    )
    printmma: Optional[bool] = Field(
        False, description="T/F: Print minimum, maximum and average"
    )
    fileout: Optional[str] = Field("history.dat", description="")
    lwxfn: Optional[bool] = Field(True, description="")
    hs: Optional[bool] = Field(False, description="significant wave height")
    tm01: Optional[bool] = Field(False, description="mean period")
    tm02: Optional[bool] = Field(False, description="zero-crossing mean period")
    klm: Optional[bool] = Field(False, description="mean wave number")
    wlm: Optional[bool] = Field(False, description="mean wave length")
    etotc: Optional[bool] = Field(False, description="Variable ETOTC")
    etots: Optional[bool] = Field(False, description="Variable ETOTS")
    dm: Optional[bool] = Field(True, description="mean wave direction")
    dspr: Optional[bool] = Field(False, description="directional spreading")
    tppd: Optional[bool] = Field(False, description="")
    tpp: Optional[bool] = Field(False, description="")
    cpp: Optional[bool] = Field(False, description="")
    wnpp: Optional[bool] = Field(False, description="peak wave number")
    cgpp: Optional[bool] = Field(False, description="peak group speed")
    kpp: Optional[bool] = Field(False, description="peak wave number")
    lpp: Optional[bool] = Field(False, description="peak")
    peakd: Optional[bool] = Field(False, description="peak direction")
    peakdspr: Optional[bool] = Field(False, description="peak directional spreading")
    dpeak: Optional[bool] = Field(False, description="")
    ubot: Optional[bool] = Field(False, description="")
    orbital: Optional[bool] = Field(False, description="")
    botexper: Optional[bool] = Field(False, description="")
    tmbot: Optional[bool] = Field(False, description="")
    ursell: Optional[bool] = Field(False, description="Ursell number")
    ufric: Optional[bool] = Field(False, description="air friction velocity")
    z0: Optional[bool] = Field(False, description="air roughness length")
    alpha_ch: Optional[bool] = Field(False, description="Charnoch coefficient for air")
    windx: Optional[bool] = Field(False, description="Wind in X direction")
    windy: Optional[bool] = Field(False, description="Wind in Y direction")
    cd: Optional[bool] = Field(False, description="Drag coefficient")
    currtx: Optional[bool] = Field(False, description="current in X direction")
    currty: Optional[bool] = Field(False, description="current in Y direction")
    watlev: Optional[bool] = Field(False, description="water level")
    watlevold: Optional[bool] = Field(
        False, description="water level at previous time step"
    )
    dep: Optional[bool] = Field(False, description="depth")
    tauw: Optional[bool] = Field(False, description="surface stress from the wave")
    tauhf: Optional[bool] = Field(False, description="high frequency surface stress")
    tautot: Optional[bool] = Field(False, description="total surface stress")
    stokessurfx: Optional[bool] = Field(
        False, description="Surface Stokes drift in X direction"
    )
    stokessurfy: Optional[bool] = Field(
        False, description="Surface Stokes drift in X direction"
    )
    stokesbarox: Optional[bool] = Field(
        False, description="Barotropic Stokes drift in X direction"
    )
    stokesbaroy: Optional[bool] = Field(
        False, description="Barotropic Stokes drift in Y direction"
    )


class Station(NamelistBaseModel):
    begtc: Optional[str] = Field(
        "19980901.000000",
        description="Start simulation time, yyyymmdd. hhmmss; must fit the simulation time otherwise no output",
    )
    deltc: Optional[str] = Field(
        "",
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )
    unitc: Optional[str] = Field("SEC", description="Unit")
    endtc: Optional[str] = Field(
        "20081101.000000", description="Stop time simulation, yyyymmdd. hhmmss"
    )
    outstyle: Optional[str] = Field(
        "STE",
        description="output option - use 'NO' to maximize efficiency during parallel run when using MPI",
    )
    fileout: Optional[str] = Field("station.dat", description="")
    loutiter: Optional[bool] = Field(False, description="")
    llouts: Optional[bool] = Field(False, description="station output flag")
    ilouts: Optional[int] = Field(1, description="Number of output stations")
    nlouts: Optional[str] = Field("", description="Name of output locations")
    iouts: Optional[int] = Field(1, description="")
    nouts: Optional[str] = Field("", description="")
    xouts: Optional[int] = Field(1, description="X-Coordinate of output locations")
    youts: Optional[int] = Field(3, description="Y-Coordinate of output locations")
    cutoff: Optional[str] = Field(
        "8*0.44",
        description="cutoff freq (Hz) for each station - consistent with buoys",
    )
    lsp1d: Optional[bool] = Field(False, description="1D spectral station output")
    lsp2d: Optional[bool] = Field(False, description="2D spectral station output")
    lsigmax: Optional[bool] = Field(
        True,
        description="Adjust the cut-freq. for the output (e.g. consistent with buoy cut-off freq.)",
    )

    @model_validator(mode="after")
    def validate_time_range(self):
        start = datetime.strptime(self.BEGTC, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.ENDTC, "%Y%m%d.%H%M%S")
        if start >= end:
            raise ValueError("BEGTC must be earlier than ENDTC")
        return self

    @model_validator(mode="after")
    def validate_station_counts(self):
        if self.ILOUTS != len(self.NLOUTS.split(",")):
            raise ValueError("ILOUTS must match the number of stations in NLOUTS")
        return self


class Petscoptions(NamelistBaseModel):
    ksptype: Optional[str] = Field("bcgs", description="")
    rtol: Optional[float] = Field(
        1e-20,
        description="the relative convergence tolerance (relative decrease in the residual norm)",
    )
    abstol: Optional[float] = Field(
        1e-20,
        description="the absolute convergence tolerance (absolute size of the residual norm)",
    )
    dtol: Optional[float] = Field(10000.0, description="the divergence tolerance")
    maxits: Optional[int] = Field(
        1000, description="maximum number of iterations to use"
    )
    initialguessnonzero: Optional[bool] = Field(
        True,
        description="Tells the iterative solver that the initial guess is nonzero; otherwise KSP assumes the initial guess is to be zero",
    )
    gmrespreallocate: Optional[bool] = Field(
        True,
        description="Causes GMRES and FGMRES to preallocate all its needed work vectors at initial setup rather than the default, which is to allocate them in chunks when needed.",
    )
    pctype: Optional[str] = Field("sor", description="")


class Wwminput(NamelistBaseModel):
    hotfile: Optional[Hotfile] = Field(default_factory=Hotfile)
    proc: Optional[Proc] = Field(default_factory=Proc)
    coupl: Optional[Coupl] = Field(default_factory=Coupl)
    grid: Optional[Grid] = Field(default_factory=Grid)
    bouc: Optional[Bouc] = Field(default_factory=Bouc)
    engs: Optional[Engs] = Field(default_factory=Engs)
    sin4: Optional[Sin4] = Field(default_factory=Sin4)
    sds4: Optional[Sds4] = Field(default_factory=Sds4)
    nums: Optional[Nums] = Field(default_factory=Nums)
    history: Optional[History] = Field(default_factory=History)
    station: Optional[Station] = Field(default_factory=Station)
    petscoptions: Optional[Petscoptions] = Field(default_factory=Petscoptions)
