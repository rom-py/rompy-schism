# This file was auto generated from a SCHISM namelist file on 2024-09-13.

from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Proc(NamelistBaseModel):
    PROCNAME: Optional[str] = Field("'Isabel'", description="Project Name")
    DIMMODE: Optional[int] = Field(
        2,
        description="Mode of run (ex: 1 = 1D, 2 = 2D) always 2D when coupled to SCHISM",
    )
    LSTEA: Optional[str] = Field("F", description="steady mode; under development")
    LQSTEA: Optional[str] = Field(
        "F",
        description="Quasi-Steady Mode; In this case WWM-II is doing subiterations defined as DELTC/NQSITER unless QSCONVI is not reached",
    )
    LSPHE: Optional[str] = Field("T", description="Spherical coordinates (lon/lat)")
    LNAUTIN: Optional[str] = Field(
        "T",
        description="Nautical convention for all inputs given in degrees (suggestion: T)",
    )
    LNAUTOUT: Optional[str] = Field("T", description="Output in Nautical convention")
    LMONO_IN: Optional[str] = Field(
        "F",
        description="For prescribing monochromatic wave height Hmono as a boundary conditions; incident wave is defined as monochromatic wave height, which is Hmono = sqrt(2) * Hs",
    )
    LMONO_OUT: Optional[str] = Field(
        "F", description="Output wave heights in terms of Lmono"
    )
    BEGTC: Optional[str] = Field(
        "'20030908.000000'",
        description="Time for start the simulation, ex:yyyymmdd. hhmmss",
    )
    DELTC: Optional[int] = Field(
        600, description="Time step (MUST match dt*nstep_wwm in SCHISM!)"
    )
    UNITC: Optional[str] = Field("'SEC'", description="Unity of time step")
    ENDTC: Optional[str] = Field(
        "'20031008.000000'",
        description="Time for stop the simulation, ex:yyyymmdd. hhmmss",
    )
    DMIN: Optional[float] = Field(
        0.01, description="Minimum water depth. This must be the same as h0 in SCHISM"
    )

    @model_validator(mode="after")
    def validate_simulation_time(self) -> "Model":
        start = datetime.strptime(self.begtc, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.endtc, "%Y%m%d.%H%M%S")
        if end <= start:
            raise ValueError("ENDTC must be later than BEGTC")
        return self


class Coupl(NamelistBaseModel):
    LCPL: Optional[str] = Field(
        "T",
        description="Couple with current model ... main switch - keep it on for SCHISM-WWM",
    )
    RADFLAG: Optional[str] = Field(
        "'LON'", description="LON: Longuet-Higgin; VOR: vortex formulation"
    )
    LETOT: Optional[str] = Field(
        "F",
        description="Option to compute the wave induced radiation stress. If .T. the radiation stress is based on the integrated wave spectrum",
    )
    NLVT: Optional[int] = Field(
        10, description="Number of vertical Layers; not used with SCHISM"
    )
    DTCOUP: Optional[float] = Field(
        600.0, description="Couple time step - not used when coupled to SCHISM"
    )


class Grid(NamelistBaseModel):
    LCIRD: Optional[str] = Field("T", description="Full circle in directional space")
    LSTAG: Optional[str] = Field(
        "F",
        description="Stagger directional bins with a half Dtheta; may use T only for regular grid to avoid char. line aligning with grid line",
    )
    MINDIR: Optional[float] = Field(
        0.0,
        description="Minimum direction for simulation (unit: degrees; nautical convention; 0: from N; 90: from E); not used if LCIRD = .T.",
    )
    MAXDIR: Optional[float] = Field(
        360.0,
        description="Maximum direction for simulation (unit: degrees); may be < MINDIR; not used if LCIRD = .T.",
    )
    MDC: Optional[int] = Field(36, description="Number of directional bins")
    FRLOW: Optional[float] = Field(
        0.04,
        description="Low frequency limit of the discrete wave period (Hz; 1/period)",
    )
    FRHIGH: Optional[float] = Field(
        1.0, description="High frequency limit of the discrete wave period."
    )
    MSC: Optional[int] = Field(36, description="Number of frequency bins")
    FILEGRID: Optional[str] = Field(
        "'hgrid_WWM.gr3'",
        description="Name of the grid file. hgridi_WWM.gr3 if IGRIDTYPE = 3 (SCHISM)",
    )
    IGRIDTYPE: Optional[int] = Field(3, description="Gridtype used.")
    LSLOP: Optional[str] = Field("F", description="Bottom Slope limiter (default=F)")
    SLMAX: Optional[float] = Field(0.2, description="Max Slope;")
    LVAR1D: Optional[str] = Field(
        "F", description="For 1d-mode if variable dx is used; not used with SCHISM"
    )
    LOPTSIG: Optional[str] = Field(
        "F",
        description="Use optimal distributions of freq. in spectral space ... fi+1 = fi * 1.1. Take care what you high freq. limit is!",
    )


class Init(NamelistBaseModel):
    FILEHOT_IN: Optional[str] = Field(
        "'hotfile_in_WWM.nc'",
        description="(Full) hot file name for input (which can be copied from FILEHOT_OUT above)",
    )
    HOTSTYLE_IN: Optional[int] = Field(
        2, description="1: binary hotfile of data as input"
    )
    IHOTPOS_IN: Optional[int] = Field(
        1, description="Position in hotfile (only for netcdf)"
    )
    MULTIPLEIN: Optional[int] = Field(
        0, description="0: read hotfile from one single file"
    )


class Hotfile(NamelistBaseModel):
    LHOTF: Optional[str] = Field("T", description="Write hotfile")
    FILEHOT_OUT: Optional[str] = Field(
        "'hotfile_out_WWM.nc'", description="name of output"
    )
    BEGTC: Optional[str] = Field(
        "'20030908.000000'",
        description="Starting time of hotfile writing. With ihot!=0 in SCHISM,",
    )
    DELTC: Optional[float] = Field(86400.0, description="time between hotfile writes")
    UNITC: Optional[str] = Field("'SEC'", description="unit used above")
    ENDTC: Optional[str] = Field(
        "'20031008.000000'",
        description="Ending time of hotfile writing (adjust with BEGTC)",
    )
    LCYCLEHOT: Optional[str] = Field("T", description="Applies only to netcdf")
    HOTSTYLE_OUT: Optional[int] = Field(
        2, description="1: binary hotfile of data as output"
    )
    MULTIPLEOUT: Optional[int] = Field(
        0, description="0: hotfile in a single file (binary or netcdf)"
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "Model":
        import datetime

        start = datetime.datetime.strptime(self.begtc, "%Y%m%d.%H%M%S")
        end = datetime.datetime.strptime(self.endtc, "%Y%m%d.%H%M%S")
        if end <= start:
            raise ValueError("ENDTC must be later than BEGTC")
        return self


class Bouc(NamelistBaseModel):
    LBCSE: Optional[str] = Field(
        "T", description="The wave boundary data is time dependent"
    )
    LBINTER: Optional[str] = Field(
        "T",
        description="Do interpolation in time if LBCSE=T (not available for quasi-steady mode within the subtime steps)",
    )
    LBCWA: Optional[str] = Field("T", description="Parametric Wave Spectra")
    LBCSP: Optional[str] = Field(
        "F",
        description="Specify (non-parametric) wave spectra, specified in 'FILEWAVE' below",
    )
    LINHOM: Optional[str] = Field("T", description="Non-uniform wave b.c. in space")
    LBSP1D: Optional[str] = Field(
        "F",
        description="1D (freq. space only) format for FILEWAVE if LBCSP=T and LINHOM=F",
    )
    LBSP2D: Optional[str] = Field(
        "F", description="2D format for FILEWAVE if LBCSP=T and LINHOM=F"
    )
    BEGTC: Optional[str] = Field(
        "'20030908.000000'",
        description="Begin time of the wave boundary file (FILEWAVE)",
    )
    DELTC: Optional[int] = Field(1, description="Time step in FILEWAVE")
    UNITC: Optional[str] = Field("'HR'", description="Unit can be HR, MIN, SEC")
    ENDTC: Optional[str] = Field("'20031008.000000'", description="End time")
    FILEBOUND: Optional[str] = Field(
        "'wwmbnd.gr3'",
        description="Boundary file defining boundary conditions and Neumann nodes.",
    )
    IBOUNDFORMAT: Optional[int] = Field(
        3, description="1 ~ WWM, 3 ~ WW3 (2D spectra in netcdf format only - LBCWA=T)."
    )
    FILEWAVE: Optional[str] = Field(
        "'bndfiles.dat'", description="Boundary file defining boundary input from WW3"
    )
    LINDSPRDEG: Optional[str] = Field(
        "F",
        description="If 1-d wave spectra are read this flag defines whether the input for the directional spreading is in degrees (true) or exponent (false)",
    )
    LPARMDIR: Optional[str] = Field(
        "F",
        description="If LPARMDIR is true then directional spreading is read from WBDS and must be in exponential format at this time, only valid for 1d Spectra",
    )
    WBHS: Optional[float] = Field(
        2.0, description="Hs at the boundary for parametric spectra"
    )
    WBSS: Optional[int] = Field(
        2,
        description="1 or -1: Pierson-Moskowitz, 2 or -2: JONSWAP, 3 or -3: all in one BIN,",
    )
    WBTP: Optional[float] = Field(
        8.0,
        description="Tp at the boundary (sec); mean or peak depending on the sign of WBSS",
    )
    WBDM: Optional[float] = Field(
        90.0, description="Avg. Wave Direction at the boundary"
    )
    WBDSMS: Optional[int] = Field(
        1, description="Directional spreading value in degrees (1) or as exponent (2)"
    )
    WBDS: Optional[float] = Field(
        10.0, description="Directional spreading at the boundary (degrees/exponent)"
    )
    WBGAUSS: Optional[float] = Field(
        0.1, description="factor for gaussian distribution if WBSS=1"
    )
    WBPKEN: Optional[float] = Field(
        3.3, description="Peak enhancement factor for Jonswap Spectra if WBSS=2"
    )

    @model_validator(mode="after")
    def validate_lbinter_lbcse(self) -> "Model":
        if self.lbinter and not self.lbcse:
            raise ValueError("LBINTER can only be True if LBCSE is True")
        return self

    @model_validator(mode="after")
    def validate_lbcwa_lbcsp(self) -> "Model":
        if self.lbcwa and self.lbcsp:
            raise ValueError("LBCWA and LBCSP cannot both be True")
        return self

    @model_validator(mode="after")
    def validate_lbsp1d_conditions(self) -> "Model":
        if self.lbsp1d and not (self.lbcsp and not self.linhom):
            raise ValueError(
                "LBSP1D can only be True if LBCSP is True and LINHOM is False"
            )
        return self

    @model_validator(mode="after")
    def validate_lbsp2d_conditions(self) -> "Model":
        if self.lbsp2d and not (self.lbcsp and not self.linhom):
            raise ValueError(
                "LBSP2D can only be True if LBCSP is True and LINHOM is False"
            )
        return self

    @model_validator(mode="after")
    def validate_time_range(self) -> "Model":
        begin = datetime.strptime(self.begtc, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.endtc, "%Y%m%d.%H%M%S")
        if end <= begin:
            raise ValueError("ENDTC must be later than BEGTC")
        return self


class Engs(NamelistBaseModel):
    MESNL: Optional[int] = Field(
        1,
        description="Nonlinear Interaction NL4 , 1 ~ on, 0 ~ off (Discrete Interaction approx.)",
    )
    MESIN: Optional[int] = Field(
        1, description="Wind input: Ardhuin et al. (1) (use LSOURCESWAM = F);"
    )
    IFRIC: Optional[int] = Field(
        1,
        description="Formulation for atmospheric boundary layer, (IFRIC = 1 for MESIN = 1, IFRIC = 4 for MESIN=3);",
    )
    MESBF: Optional[int] = Field(
        1,
        description="Bottom friction: 1 - JONSWAP (Default); 2 - Madsen et al. (1989); 3 - SHOWEX",
    )
    FRICC: Optional[float] = Field(
        0.067,
        description="if MESBF=1: JONSWAP bottom friction coefficient [0.038,0.067]. If MESBF=2: physical bottom roughness (ignored if given in rough.gr3). If MESBF=3: D50 (if negative read from SHOWEX_D50.gr3)",
    )
    MESBR: Optional[int] = Field(
        1, description="Shallow water wave breaking; 0: off; 1: on"
    )
    IBREAK: Optional[int] = Field(
        1, description="Wave breaking formulation: 1 - Battjes and Janssen (1978)"
    )
    ICRIT: Optional[int] = Field(
        1,
        description="Wave breaking criterion: 1   - Constant breaker index (gamma) or gamma_TG defined with BRCR",
    )
    BRCR: Optional[float] = Field(
        0.78,
        description="either gamma, default is 0.73 for IBREAK=1,5 or gamma_TG, default is 0.42 for IBREAK=2,3 or biphase_ref, default is -4pi/9 = -1.3963 for IBREAK=4",
    )
    a_BRCR: Optional[float] = Field(0.76, description="cf ICRIT = 4, 5")
    b_BRCR: Optional[float] = Field(0.29, description="cf ICRIT = 4, 5")
    min_BRCR: Optional[float] = Field(0.25, description="cf ICRIT = 4, 5")
    max_BRCR: Optional[float] = Field(0.8, description="cf ICRIT = 4, 5")
    a_BIPH: Optional[float] = Field(
        0.2, description="Biphase coefficient, default 0.2 (intended for IBREAK=3)"
    )
    BR_COEF_METHOD: Optional[int] = Field(
        1, description="Method for the breaking coefficient: 1 - constant, 2 - adaptive"
    )
    B_ALP: Optional[float] = Field(
        0.5, description="breaking coefficient. If BR_COEF_METHOD = 2, B_ALP ~ 40"
    )
    ZPROF_BREAK: Optional[int] = Field(
        2,
        description="Vertical distribution function of wave breaking source term, only used in 3D run",
    )
    BC_BREAK: Optional[int] = Field(
        1, description="Apply depth-limited breaking at the boundaries: 1 - On; 0 - Off"
    )
    IROLLER: Optional[int] = Field(
        0,
        description="Wave roller model (e.g., see Uchiyama et al., 2010): 1 - On; 0 - Off; not used at the moment",
    )
    ALPROL: Optional[float] = Field(
        0.85,
        description="Alpha coefficient for the wave roller model (between 0 and 1): 1 - full conversion; 0 - no energy transferred to the roller",
    )
    MEVEG: Optional[int] = Field(
        0, description="Vegetation on/off. If on, isav must = 1 in param.nml"
    )
    LMAXETOT: Optional[str] = Field(
        "T",
        description="Limit shallow water wave height by wave breaking limiter (default=T)",
    )
    MESDS: Optional[int] = Field(
        1,
        description="Formulation for the whitecapping source function; same value as MESIN",
    )
    MESTR: Optional[int] = Field(
        1,
        description="Formulation for the triad 3 wave interactions (MESTR = 0 (off), MESTR = 1 (Lumped Triad Approx. (LTA)), MESTR = 2 (corrected version of LTA by Salmon et al. (2016)))",
    )
    TRICO: Optional[float] = Field(
        0.1, description="proportionality const. (\alpha_EB); default is 0.1"
    )
    TRIRA: Optional[float] = Field(
        2.5,
        description="ratio of max. freq. considered in triads over mean freq.; 2.5 is suggested",
    )
    TRIURS: Optional[float] = Field(
        0.1,
        description="critical Ursell number; if Ursell # < TRIURS; triads are not computed",
    )

    @model_validator(mode="after")
    def validate_mesin_lsourceswam(self):
        if self.mesin == 1 and self.lsourceswam:
            raise ValueError("When MESIN=1, LSOURCESWAM should be False")
        return self

    @model_validator(mode="after")
    def validate_ifric_mesin(self):
        if self.mesin == 1 and self.ifric != 1:
            raise ValueError("When MESIN=1, IFRIC should be 1")
        if self.mesin == 3 and self.ifric != 4:
            raise ValueError("When MESIN=3, IFRIC should be 4")
        return self

    @model_validator(mode="after")
    def validate_fricc_mesbf(self):
        if self.mesbf == 1 and not 0.038 <= self.fricc <= 0.067:
            raise ValueError("When MESBF=1, FRICC should be between 0.038 and 0.067")
        return self

    @model_validator(mode="after")
    def validate_mesds_mesin(self):
        if self.mesds != self.mesin:
            raise ValueError("MESDS should have the same value as MESIN")
        return self


class Nums(NamelistBaseModel):
    ICOMP: Optional[int] = Field(3, description="")
    AMETHOD: Optional[int] = Field(7, description="")
    SMETHOD: Optional[int] = Field(1, description="")
    DMETHOD: Optional[int] = Field(2, description="")
    RTHETA: Optional[float] = Field(
        0.5,
        description="Weighing factor for DMETHOD = 1, not really useful since Crank Nicholson integration can only be monotone for CFL .le. 2",
    )
    LITERSPLIT: Optional[str] = Field(
        "F",
        description="T: double Strang split; F: simple split (more efficient). Default: F",
    )
    LFILTERTH: Optional[str] = Field("F", description="")
    MAXCFLTH: Optional[float] = Field(
        1.0, description="Max Cfl in Theta space; used only if LFILTERTH=T"
    )
    FMETHOD: Optional[int] = Field(1, description="")
    LFILTERSIG: Optional[str] = Field(
        "F", description="Limit the advection velocity in freq. space (usually F)"
    )
    MAXCFLSIG: Optional[float] = Field(
        1.0, description="Max Cfl in freq. space; used only if LFILTERSIG=T"
    )
    LLIMT: Optional[str] = Field(
        "T",
        description="Switch on/off Action limiter, Action limiter must mostly be turned on.",
    )
    MELIM: Optional[int] = Field(1, description="Formulation for the action limiter")
    LIMFAK: Optional[float] = Field(
        0.1,
        description="Proportionality coefficient for the action limiter MAX_DAC_DT = Limfak * Limiter; see notes above for value",
    )
    LDIFR: Optional[str] = Field(
        "F",
        description="Use phase decoupled diffraction approximation according to Holthuijsen et al. (2003) (usually T; if crash, use F)",
    )
    IDIFFR: Optional[int] = Field(
        1,
        description="Extended WAE accounting for higher order effects WAE becomes nonlinear; 1: Holthuijsen et al. ; 2: Liau et al. ; 3: Toledo et al. (in preparation)",
    )
    LCONV: Optional[str] = Field(
        "F",
        description="Estimate convergence criterian and write disk (quasi-steady - qstea.out)",
    )
    LCFL: Optional[str] = Field(
        "F", description="Write out CFL numbers; use F to save time"
    )
    NQSITER: Optional[int] = Field(
        1,
        description="# of quasi-steady (Q-S) sub-divisions within each WWM time step (trial and errors)",
    )
    QSCONV1: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill abs. wave height criteria EPSH1",
    )
    QSCONV2: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill rel. wave height criteria EPSH2",
    )
    QSCONV3: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill sum. rel. wave action criteria EPSH3",
    )
    QSCONV4: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill rel. avg. wave steepness criteria EPSH4",
    )
    QSCONV5: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave period criteria EPSH5",
    )
    LEXPIMP: Optional[str] = Field(
        "F",
        description="Use implicit schemes for freq. lower than given below by FREQEXP; used only if ICOMP=0",
    )
    FREQEXP: Optional[float] = Field(
        0.1,
        description="Minimum frequency for explicit schemes; only used if LEXPIMP=T and ICOMP=0",
    )
    EPSH1: Optional[float] = Field(
        0.01,
        description="Convergence criteria for rel. wave height ! EPSH1 < CONVK1 = REAL(ABS(HSOLD(IP)-HS2)/HS2)",
    )
    EPSH2: Optional[float] = Field(
        0.01,
        description="Convergence criteria for abs. wave height ! EPSH2 < CONVK2 = REAL(ABS(HS2-HSOLD(IP)))",
    )
    EPSH3: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. sum of wave action ! EPSH3 < CONVK3 = REAL(ABS(SUMACOLD(IP)-SUMAC)/SUMAC)",
    )
    EPSH4: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. avg. wave steepness criteria ! EPSH4 < CONVK4 = REAL(ABS(KHS2-KHSOLD(IP))/KHSOLD(IP))",
    )
    EPSH5: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. avg. waveperiod ! EPSH5 < REAL(ABS(TM02-TM02OLD(IP))/TM02OLD(IP))",
    )
    LVECTOR: Optional[str] = Field(
        "F",
        description="Use optmized propagation routines for large high performance computers e.g. at least more than 128 CPU. Try LVECTOR=F first.",
    )
    IVECTOR: Optional[int] = Field(
        2, description="USed if LVECTOR=T; Different flavours of communications"
    )
    LADVTEST: Optional[str] = Field(
        "F",
        description="for testing the advection schemes, testcase will be added soon",
    )
    LCHKCONV: Optional[str] = Field(
        "F",
        description="needs to set to .true. for quasi-steady mode. in order to compute the QSCONVi criteria and check them",
    )
    DTMIN_DYN: Optional[float] = Field(
        1.0,
        description="min. time step (sec?) for dynamic integration, this controls in SMETHOD the smallest time step for the triads, DT = 1.s is found to work well.",
    )
    NDYNITER: Optional[list] = Field(
        [100, ""],
        description="max. iteration for dyn. scheme afterwards the limiter is applied in the last step, for SMETHOD .eq. this controls the integration of the triad interaction terms, which is done dynamically.",
    )
    DTMIN_SIN: Optional[float] = Field(
        1.0,
        description="min. time steps for the full fractional step method, where each source term is integrated with its own fractional step",
    )
    DTMIN_SNL4: Optional[float] = Field(1.0, description="")
    DTMIN_SDS: Optional[float] = Field(1.0, description="")
    DTMIN_SNL3: Optional[float] = Field(1.0, description="")
    DTMIN_SBR: Optional[float] = Field(0.1, description="")
    DTMIN_SBF: Optional[float] = Field(1.0, description="")
    NDYNITER_SIN: Optional[list] = Field(
        [10, ""],
        description="max. iterations for each source term in the fractional step approach.",
    )
    NDYNITER_SNL4: Optional[list] = Field([10, ""], description="")
    NDYNITER_SDS: Optional[list] = Field([10, ""], description="")
    NDYNITER_SBR: Optional[list] = Field([10, ""], description="")
    NDYNITER_SNL3: Optional[list] = Field([10, ""], description="")
    NDYNITER_SBF: Optional[list] = Field([10, ""], description="")
    LSOUBOUND: Optional[str] = Field(
        "F",
        description="Do source terms on boundary, this is possible as long only dissipative processes are governing the spectral evolution, otherwise, with wind u will get the max. possible wave height",
    )
    WAE_SOLVERTHR: Optional[list] = Field(
        ["1.e-6", ""],
        description="Threshold for the Block-Jacobi or Block-Gauss-Seider solver",
    )
    MAXITER: Optional[list] = Field([1000, ""], description="Max. number of iterations")
    PMIN: Optional[list] = Field(
        [1.0, ""], description="Max. percentage of non-converged grid points"
    )
    LNANINFCHK: Optional[list] = Field(
        ["F", ""],
        description="Check for NaN and INF; usually turned off for efficiency",
    )
    LZETA_SETUP: Optional[list] = Field(
        ["F", ""], description="Compute wave setup (simple momentum eq.)"
    )
    ZETA_METH: Optional[list] = Field(
        [0, ""], description="Method for wave setup, Mathieu please explain!"
    )
    LSOURCESWAM: Optional[list] = Field(
        ["F", ""], description="Use ECMWF WAM formualtion for deep water physics."
    )
    BLOCK_GAUSS_SEIDEL: Optional[list] = Field(
        ["T", ""], description="Use the Gauss Seidel on each"
    )
    LNONL: Optional[str] = Field(
        "F", description="Solve the nonlinear system using simpler algorithm (Patankar)"
    )
    ASPAR_LOCAL_LEVEL: Optional[int] = Field(0, description="Aspar locality level")
    L_SOLVER_NORM: Optional[str] = Field(
        "F", description="Compute solver norm ||A*x-b|| as termination"
    )
    LACCEL: Optional[str] = Field("F", description="")

    @model_validator(mode="after")
    def validate_ivector_lvector(self):
        if self.lvector and self.ivector not in range(1, 7):
            raise ValueError("When LVECTOR is True, IVECTOR must be between 1 and 6")
        return self


class History(NamelistBaseModel):
    BEGTC: Optional[str] = Field(
        "'20030908.000000'", description="Start output time, yyyymmdd. hhmmss;"
    )
    DELTC: Optional[int] = Field(
        1,
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )
    UNITC: Optional[str] = Field("'SEC'", description="Unit")
    ENDTC: Optional[str] = Field(
        "'20031008.000000'", description="Stop time output, yyyymmdd. hhmmss"
    )
    DEFINETC: Optional[int] = Field(
        86400, description="Time scoop (sec) for history files"
    )
    OUTSTYLE: Optional[str] = Field(
        "'NO'", description="output option - use 'NO' for no output"
    )
    MULTIPLEOUT: Optional[int] = Field(
        0, description="0: output in a single netcdf file"
    )
    USE_SINGLE_OUT: Optional[str] = Field(
        "T", description="T: Use single precision in the"
    )
    PARAMWRITE: Optional[str] = Field(
        "T", description="T: Write the physical parametrization"
    )
    GRIDWRITE: Optional[str] = Field(
        "T", description="T/F: Write the grid in the netcdf history file (default T)"
    )
    PRINTMMA: Optional[str] = Field(
        "F", description="T/F: Print minimum, maximum and average"
    )
    FILEOUT: Optional[str] = Field("'wwm_hist.dat'", description="")
    HS: Optional[str] = Field("F", description="significant wave height")
    TM01: Optional[str] = Field("F", description="mean period")
    TM02: Optional[str] = Field("F", description="zero-crossing mean period")
    KLM: Optional[str] = Field("F", description="mean wave number")
    WLM: Optional[str] = Field("F", description="mean wave length")
    ETOTC: Optional[str] = Field("F", description="Variable ETOTC")
    ETOTS: Optional[str] = Field("F", description="Variable ETOTS")
    DM: Optional[str] = Field("F", description="mean wave direction")
    DSPR: Optional[str] = Field("F", description="directional spreading")
    TPPD: Optional[str] = Field(
        "F", description="direaction of the peak ... check source code"
    )
    TPP: Optional[str] = Field("F", description="peak period")
    CPP: Optional[str] = Field("F", description="peak phase vel.")
    WNPP: Optional[str] = Field("F", description="peak wave number")
    CGPP: Optional[str] = Field("F", description="peak group speed")
    KPP: Optional[str] = Field("F", description="peak wave number")
    LPP: Optional[str] = Field("F", description="peak wave length")
    PEAKD: Optional[str] = Field("F", description="peak direction")
    PEAKDSPR: Optional[str] = Field("F", description="peak directional spreading")
    DPEAK: Optional[str] = Field("F", description="peak direction")
    UBOT: Optional[str] = Field("F", description="bottom exc. vel.")
    ORBITAL: Optional[str] = Field("F", description="bottom orbital vel.")
    BOTEXPER: Optional[str] = Field("F", description="bottom exc.")
    TMBOT: Optional[str] = Field("F", description="bottom period")
    URSELL: Optional[str] = Field("F", description="Ursell number")
    UFRIC: Optional[str] = Field("F", description="air friction velocity")
    Z0: Optional[str] = Field("F", description="air roughness length")
    ALPHA_CH: Optional[str] = Field("F", description="Charnoch coefficient for air")
    WINDX: Optional[str] = Field("F", description="Wind in X direction")
    WINDY: Optional[str] = Field("F", description="Wind in Y direction")
    CD: Optional[str] = Field("F", description="Drag coefficient")
    CURRTX: Optional[str] = Field("F", description="current in X direction")
    CURRTY: Optional[str] = Field("F", description="current in Y direction")
    WATLEV: Optional[str] = Field("F", description="water level")
    WATLEVOLD: Optional[str] = Field(
        "F", description="water level at previous time step"
    )
    DEPDT: Optional[str] = Field("F", description="change of water level in time")
    DEP: Optional[str] = Field("F", description="depth")
    TAUW: Optional[str] = Field("F", description="surface stress from the wave")
    TAUHF: Optional[str] = Field("F", description="high frequency surface stress")
    TAUTOT: Optional[str] = Field("F", description="total surface stress")
    STOKESSURFX: Optional[str] = Field(
        "F", description="Surface Stokes drift in X direction"
    )
    STOKESSURFY: Optional[str] = Field(
        "F", description="Surface Stokes drift in X direction"
    )
    STOKESBAROX: Optional[str] = Field(
        "F", description="Barotropic Stokes drift in X direction"
    )
    STOKESBAROY: Optional[str] = Field(
        "F", description="Barotropic Stokes drift in Y direction"
    )
    RSXX: Optional[str] = Field("F", description="RSXX potential of LH")
    RSXY: Optional[str] = Field("F", description="RSXY potential of LH")
    RSYY: Optional[str] = Field("F", description="RSYY potential of LH")
    CFL1: Optional[str] = Field("F", description="CFL number 1")
    CFL2: Optional[str] = Field("F", description="CFL number 2")
    CFL3: Optional[str] = Field("F", description="CFL number 3")


class Station(NamelistBaseModel):
    BEGTC: Optional[str] = Field(
        "'20030908.000000'",
        description="Start simulation time, yyyymmdd. hhmmss; must fit the simulation time otherwise no output",
    )
    DELTC: Optional[int] = Field(
        600,
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )
    UNITC: Optional[str] = Field("'SEC'", description="Unit")
    ENDTC: Optional[str] = Field(
        "'20031008.000000'", description="Stop time simulation, yyyymmdd. hhmmss"
    )
    DEFINETC: Optional[int] = Field(
        86400, description="Time for definition of station files"
    )
    OUTSTYLE: Optional[str] = Field("'NO'", description="output option")
    MULTIPLEOUT: Optional[int] = Field(
        0, description="0: output in a single netcdf file"
    )
    USE_SINGLE_OUT: Optional[str] = Field(
        "T", description="T: Use single precision in the"
    )
    PARAMWRITE: Optional[str] = Field(
        "T", description="T: Write the physical parametrization"
    )
    FILEOUT: Optional[str] = Field("'wwm_sta.dat'", description="not used")
    LOUTITER: Optional[str] = Field("F", description="")
    IOUTS: Optional[list] = Field([15, ""], description="")
    NOUTS: Optional[list] = Field(
        [
            "'P-1'",
            "'P-2'",
            "'P-3'",
            "'P-4'",
            "'P-5'",
            "'P-6'",
            "'P-7'",
            "'P-8'",
            "'P-9'",
            "'P-10'",
            "'P-11'",
            "'P-12'",
            "'P-13'",
            "'P-14'",
            "'P-15'",
        ],
        description="",
    )
    XOUTS: Optional[list] = Field(
        [-76.046, -76.778, -75.81, -75.72, -74.842, ""], description=""
    )
    YOUTS: Optional[list] = Field([39.152, 38.556, 38.033, 37.551, ""], description="")
    CUTOFF: Optional[str] = Field(
        "15*0.44",
        description="cutoff freq (Hz) for each station - consistent with buoys",
    )
    LSP1D: Optional[str] = Field("T", description="1D spectral station output")
    LSP2D: Optional[str] = Field("F", description="2D spectral station output")
    LSIGMAX: Optional[str] = Field(
        "T",
        description="Adjust the cut-freq. for the output (e.g. consistent with buoy cut-off freq.)",
    )
    AC: Optional[str] = Field("F", description="spectrum")
    WK: Optional[str] = Field("F", description="variable WK")
    ACOUT_1D: Optional[str] = Field("F", description="variable ACOUT_1D")
    ACOUT_2D: Optional[str] = Field("F", description="variable ACOUT_2D")
    HS: Optional[str] = Field("F", description="significant wave height")
    TM01: Optional[str] = Field("F", description="mean period")
    TM02: Optional[str] = Field("F", description="zero-crossing mean period")
    KLM: Optional[str] = Field("F", description="mean wave number")
    WLM: Optional[str] = Field("F", description="mean wave length")
    ETOTC: Optional[str] = Field("F", description="Variable ETOTC")
    ETOTS: Optional[str] = Field("F", description="Variable ETOTS")
    DM: Optional[str] = Field("F", description="mean wave direction")
    DSPR: Optional[str] = Field("F", description="directional spreading")
    TPPD: Optional[str] = Field("F", description="Discrete Peak Period")
    TPP: Optional[str] = Field("F", description="Peak Period")
    CPP: Optional[str] = Field("F", description="")
    WNPP: Optional[str] = Field("F", description="peak wave number")
    CGPP: Optional[str] = Field("F", description="peak group speed")
    KPP: Optional[str] = Field("F", description="peak wave number")
    LPP: Optional[str] = Field("F", description="peak")
    PEAKD: Optional[str] = Field("F", description="peak direction")
    PEAKDSPR: Optional[str] = Field("F", description="peak directional spreading")
    DPEAK: Optional[str] = Field("F", description="")
    UBOT: Optional[str] = Field("F", description="")
    ORBITAL: Optional[str] = Field("F", description="")
    BOTEXPER: Optional[str] = Field("F", description="")
    TMBOT: Optional[str] = Field("F", description="")
    URSELL: Optional[str] = Field("F", description="Ursell number")
    UFRIC: Optional[str] = Field("F", description="air friction velocity")
    Z0: Optional[str] = Field("F", description="air roughness length")
    ALPHA_CH: Optional[str] = Field("F", description="Charnoch coefficient for air")
    WINDX: Optional[str] = Field("F", description="Wind in X direction")
    WINDY: Optional[str] = Field("F", description="Wind in Y direction")
    CD: Optional[str] = Field("F", description="Drag coefficient")
    CURRTX: Optional[str] = Field("F", description="current in X direction")
    CURRTY: Optional[str] = Field("F", description="current in Y direction")
    WATLEV: Optional[str] = Field("F", description="water level")
    WATLEVOLD: Optional[str] = Field(
        "F", description="water level at previous time step"
    )
    DEPDT: Optional[str] = Field("F", description="change of water level in time")
    DEP: Optional[str] = Field("F", description="depth")
    TAUW: Optional[str] = Field("F", description="surface stress from the wave")
    TAUHF: Optional[str] = Field("F", description="high frequency surface stress")
    TAUTOT: Optional[str] = Field("F", description="total surface stress")
    STOKESSURFX: Optional[str] = Field(
        "F", description="Surface Stokes drift in X direction"
    )
    STOKESSURFY: Optional[str] = Field(
        "F", description="Surface Stokes drift in X direction"
    )
    STOKESBAROX: Optional[str] = Field(
        "F", description="Barotropic Stokes drift in X direction"
    )
    STOKESBAROY: Optional[str] = Field(
        "F", description="Barotropic Stokes drift in Y direction"
    )
    RSXX: Optional[str] = Field("F", description="RSXX potential of LH")
    RSXY: Optional[str] = Field("F", description="RSXY potential of LH")
    RSYY: Optional[str] = Field("F", description="RSYY potential of LH")
    CFL1: Optional[str] = Field("F", description="CFL number 1")
    CFL2: Optional[str] = Field("F", description="CFL number 2")
    CFL3: Optional[str] = Field("F", description="CFL number 3")

    @model_validator(mode="after")
    def validate_simulation_time(self) -> "Model":
        start = datetime.strptime(self.begtc, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.endtc, "%Y%m%d.%H%M%S")
        if end <= start:
            raise ValueError("ENDTC must be later than BEGTC")
        return self

    @model_validator(mode="after")
    def validate_nouts_length(self) -> "Model":
        if len(self.nouts) != self.iouts:
            raise ValueError("Length of NOUTS must match IOUTS")
        return self

    @model_validator(mode="after")
    def validate_xouts_length(self) -> "Model":
        if len(self.xouts) != self.iouts:
            raise ValueError("Length of XOUTS must match IOUTS")
        return self

    @model_validator(mode="after")
    def validate_youts_length(self) -> "Model":
        if len(self.youts) != self.iouts:
            raise ValueError("Length of YOUTS must match IOUTS")
        return self

    @model_validator(mode="after")
    def validate_cutoff_length(self) -> "Model":
        if len(self.cutoff) != self.iouts:
            raise ValueError("Length of CUTOFF must match IOUTS")
        return self


class Petscoptions(NamelistBaseModel):
    KSPTYPE: Optional[str] = Field("'LGMRES'", description="")
    RTOL: Optional[str] = Field(
        "1.E-20",
        description="the relative convergence tolerance (relative decrease in the residual norm)",
    )
    ABSTOL: Optional[str] = Field(
        "1.E-20",
        description="the absolute convergence tolerance (absolute size of the residual norm)",
    )
    DTOL: Optional[float] = Field(10000.0, description="the divergence tolerance")
    MAXITS: Optional[int] = Field(
        1000, description="maximum number of iterations to use"
    )
    INITIALGUESSNONZERO: Optional[str] = Field(
        "F",
        description="Tells the iterative solver that the initial guess is nonzero; otherwise KSP assumes the initial guess is to be zero",
    )
    GMRESPREALLOCATE: Optional[str] = Field(
        "T",
        description="Causes GMRES and FGMRES to preallocate all its needed work vectors at initial setup rather than the default, which is to allocate them in chunks when needed.",
    )
    PCTYPE: Optional[str] = Field("'SOR'", description="")


class Nesting(NamelistBaseModel):
    ListBEGTC: Optional[str] = Field("''", description="")
    ListDELTC: Optional[str] = Field("ZERO", description="")
    ListUNITC: Optional[str] = Field("''", description="")
    ListENDTC: Optional[str] = Field("''", description="")
    ListIGRIDTYPE: Optional[int] = Field(0, description="")
    ListFILEGRID: Optional[str] = Field("''", description="")
    ListFILEBOUND: Optional[str] = Field("''", description="")
    ListPrefix: Optional[str] = Field("''", description="")


class Wwminput(NamelistBaseModel):
    proc: Optional[Proc] = Field(default_factory=Proc)
    coupl: Optional[Coupl] = Field(default_factory=Coupl)
    grid: Optional[Grid] = Field(default_factory=Grid)
    init: Optional[Init] = Field(default_factory=Init)
    hotfile: Optional[Hotfile] = Field(default_factory=Hotfile)
    bouc: Optional[Bouc] = Field(default_factory=Bouc)
    engs: Optional[Engs] = Field(default_factory=Engs)
    nums: Optional[Nums] = Field(default_factory=Nums)
    history: Optional[History] = Field(default_factory=History)
    station: Optional[Station] = Field(default_factory=Station)
    petscoptions: Optional[Petscoptions] = Field(default_factory=Petscoptions)
    nesting: Optional[Nesting] = Field(default_factory=Nesting)
