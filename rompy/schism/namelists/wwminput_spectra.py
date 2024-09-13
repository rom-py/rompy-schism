# This file was auto generated from a SCHISM namelist file on 2024-09-13.

from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Hotfile(NamelistBaseModel):
    LHOTF: Optional[str] = Field(
        "T", description="Boolean flag to enable writing of hotfile output."
    )
    BEGTC: Optional[str] = Field(
        "'19980901.000000'",
        description="Starting time of hotfile writing in the format 'YYYYMMDD.HHMMSS'.",
    )
    DELTC: Optional[int] = Field(
        3600,
        description="Time interval between hotfile writes, specified in the units defined by UNITC.",
    )
    UNITC: Optional[str] = Field(
        "'SEC'",
        description="Unit of time used for DELTC, currently set to 'SEC' for seconds.",
    )
    ENDTC: Optional[str] = Field(
        "'19980901.060000'",
        description="Ending time of hotfile writing in the format 'YYYYMMDD.HHMMSS'.",
    )
    LCYCLEHOT: Optional[str] = Field(
        "T",
        description="Boolean flag to determine hotfile record behavior for netCDF. If True, hotfile contains 2 last records (1st record is most recent). If False, hotfile contains N records if N outputs have been done. For binary, only one record is used regardless of this setting.",
    )
    HOTSTYLE_OUT: Optional[int] = Field(
        2,
        description="Integer flag to specify the output format of the hotfile. 1 for binary, 2 for netCDF (default).",
    )
    MULTIPLEOUT: Optional[int] = Field(
        0,
        description="Integer flag to determine the output file structure. 0 for a single file (binary or netCDF) using MPI_REDUCE, 1 for separate files associated with each process.",
    )
    FILEHOT_OUT: Optional[str] = Field(
        "'hotfile_out_WWM.nc'", description="Name of the hotfile output file."
    )

    @field_validator("LHOTF")
    @classmethod
    def validate_lhotf(cls, v: bool) -> bool:
        return v

    @field_validator("BEGTC")
    @classmethod
    def validate_begtc(cls, v: str) -> str:
        if not re.match(r"\d{8}\.\d{6}", v):
            raise ValueError("BEGTC must be in format YYYYMMDD.HHMMSS")
        return v

    @field_validator("DELTC")
    @classmethod
    def validate_deltc(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("DELTC must be a positive integer")
        return v

    @field_validator("UNITC")
    @classmethod
    def validate_unitc(cls, v: str) -> str:
        if v.upper() != "SEC":
            raise ValueError("UNITC must be SEC")
        return v.upper()

    @field_validator("ENDTC")
    @classmethod
    def validate_endtc(cls, v: str) -> str:
        if not re.match(r"\d{8}\.\d{6}", v):
            raise ValueError("ENDTC must be in format YYYYMMDD.HHMMSS")
        return v

    @field_validator("LCYCLEHOT")
    @classmethod
    def validate_lcyclehot(cls, v: bool) -> bool:
        return v

    @field_validator("HOTSTYLE_OUT")
    @classmethod
    def validate_hotstyle_out(cls, v: int) -> int:
        if v not in [1, 2]:
            raise ValueError("HOTSTYLE_OUT must be 1 or 2")
        return v

    @field_validator("MULTIPLEOUT")
    @classmethod
    def validate_multipleout(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError("MULTIPLEOUT must be 0 or 1")
        return v

    @field_validator("FILEHOT_OUT")
    @classmethod
    def validate_filehot_out(cls, v: str) -> str:
        if not v.endswith(".nc"):
            raise ValueError("FILEHOT_OUT must end with .nc")
        return v

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
    PROCNAME: Optional[str] = Field(
        "'limon'", description="Project Name for the simulation"
    )
    DIMMODE: Optional[int] = Field(
        2, description="Dimension mode of the run. Always 2 when coupled to SCHISM"
    )
    LSTEA: Optional[str] = Field(
        "F", description="Flag for steady mode (under development)"
    )
    LQSTEA: Optional[str] = Field(
        "F",
        description="Flag for Quasi-Steady Mode. If True, WWM-II performs subiterations defined as DELTC/NQSITER unless QSCONVI is reached",
    )
    LSPHE: Optional[str] = Field(
        "F", description="Flag for using spherical coordinates (longitude/latitude)"
    )
    LNAUTIN: Optional[str] = Field(
        "T",
        description="Flag for using nautical convention for all inputs given in degrees. If True, 0 is from north, 90 is from east. If False, mathematical convention is used (0: to east; 90: to north)",
    )
    LMONO_IN: Optional[str] = Field("F", description="Flag for mono input")
    LMONO_OUT: Optional[str] = Field("F", description="Flag for mono output")
    LNAUTOUT: Optional[str] = Field(
        "T", description="Flag for nautical output of all quantities in degrees"
    )
    BEGTC: Optional[str] = Field(
        "'19980901.000000'",
        description="Start time of the simulation in format 'yyyymmdd.hhmmss'",
    )
    DELTC: Optional[int] = Field(
        5, description="Time step in seconds. Must match dt*nstep_wwm in SCHISM"
    )
    UNITC: Optional[str] = Field("'SEC'", description="Unit of time step")
    ENDTC: Optional[str] = Field(
        "'19980901.060000'",
        description="End time of the simulation in format 'yyyymmdd.hhmmss'",
    )
    DMIN: Optional[float] = Field(
        0.01, description="Minimum water depth. Must be the same as h0 in SCHISM"
    )

    @field_validator("PROCNAME")
    @classmethod
    def validate_procname(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("PROCNAME must be a non-empty string")
        return v

    @field_validator("DIMMODE")
    @classmethod
    def validate_dimmode(cls, v):
        if v != 2:
            raise ValueError("DIMMODE must be 2 when coupled to SCHISM")
        return v

    @field_validator("LSTEA")
    @classmethod
    def validate_lstea(cls, v):
        return v

    @field_validator("LQSTEA")
    @classmethod
    def validate_lqstea(cls, v):
        return v

    @field_validator("LSPHE")
    @classmethod
    def validate_lsphe(cls, v):
        return v

    @field_validator("LNAUTIN")
    @classmethod
    def validate_lnautin(cls, v):
        return v

    @field_validator("LMONO_IN")
    @classmethod
    def validate_lmono_in(cls, v):
        return v

    @field_validator("LMONO_OUT")
    @classmethod
    def validate_lmono_out(cls, v):
        return v

    @field_validator("LNAUTOUT")
    @classmethod
    def validate_lnautout(cls, v):
        return v

    @field_validator("BEGTC")
    @classmethod
    def validate_begtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
        except ValueError:
            raise ValueError("BEGTC must be in format yyyymmdd.hhmmss")
        return v

    @field_validator("DELTC")
    @classmethod
    def validate_deltc(cls, v):
        if v <= 0:
            raise ValueError("DELTC must be positive")
        return v

    @field_validator("UNITC")
    @classmethod
    def validate_unitc(cls, v):
        if v.upper() != "SEC":
            raise ValueError("UNITC must be SEC")
        return v.upper()

    @field_validator("ENDTC")
    @classmethod
    def validate_endtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
        except ValueError:
            raise ValueError("ENDTC must be in format yyyymmdd.hhmmss")
        return v

    @field_validator("DMIN")
    @classmethod
    def validate_dmin(cls, v):
        if v < 0:
            raise ValueError("DMIN must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_time_range(self):
        start = datetime.strptime(self.BEGTC, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.ENDTC, "%Y%m%d.%H%M%S")
        if end <= start:
            raise ValueError("ENDTC must be later than BEGTC")
        return self


class Coupl(NamelistBaseModel):
    LCPL: Optional[str] = Field(
        "T",
        description="Main switch for coupling with current model. Should be kept on for SCHISM-WWM coupling.",
    )
    LROMS: Optional[str] = Field(
        "F", description="Switch for ROMS coupling. Should be set to False."
    )
    LTIMOR: Optional[str] = Field(
        "F", description="Switch for TIMOR coupling. Should be set to False."
    )
    LSHYFEM: Optional[str] = Field(
        "F", description="Switch for SHYFEM coupling. Should be set to False."
    )
    RADFLAG: Optional[str] = Field(
        "'LON'",
        description="Flag for radiation stress calculation method. Currently set to 'LON'.",
    )
    LETOT: Optional[str] = Field(
        "F",
        description="Option to compute wave-induced radiation stress. If True, radiation stress is based on the integrated wave spectrum. If False (recommended), it's estimated based on the directional spectra itself as given in Roland et al. (2008). False is preferred to preserve spectral information.",
    )
    NLVT: Optional[int] = Field(
        10, description="Number of vertical layers. Not used with SCHISM."
    )
    DTCOUP: Optional[float] = Field(
        600.0,
        description="Coupling time step in seconds. Not used when coupled to SCHISM.",
    )

    @field_validator("LCPL")
    @classmethod
    def validate_lcpl(cls, v):
        if not isinstance(v, bool):
            raise ValueError("LCPL must be a boolean")
        return v

    @field_validator("LROMS")
    @classmethod
    def validate_lroms(cls, v):
        if not isinstance(v, bool):
            raise ValueError("LROMS must be a boolean")
        if v:
            raise ValueError("LROMS should be set to False")
        return v

    @field_validator("LTIMOR")
    @classmethod
    def validate_ltimor(cls, v):
        if not isinstance(v, bool):
            raise ValueError("LTIMOR must be a boolean")
        if v:
            raise ValueError("LTIMOR should be set to False")
        return v

    @field_validator("LSHYFEM")
    @classmethod
    def validate_lshyfem(cls, v):
        if not isinstance(v, bool):
            raise ValueError("LSHYFEM must be a boolean")
        if v:
            raise ValueError("LSHYFEM should be set to False")
        return v

    @field_validator("RADFLAG")
    @classmethod
    def validate_radflag(cls, v):
        if not isinstance(v, str):
            raise ValueError("RADFLAG must be a string")
        return v

    @field_validator("LETOT")
    @classmethod
    def validate_letot(cls, v):
        if not isinstance(v, bool):
            raise ValueError("LETOT must be a boolean")
        if v:
            import warnings

            warnings.warn("Setting LETOT to True is only for testing and developers.")
        return v

    @field_validator("NLVT")
    @classmethod
    def validate_nlvt(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("NLVT must be a positive integer")
        return v

    @field_validator("DTCOUP")
    @classmethod
    def validate_dtcoup(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("DTCOUP must be a positive number")
        return v


class Grid(NamelistBaseModel):
    LCIRD: Optional[str] = Field(
        "T",
        description="Boolean flag to use full circle in directional space. If True, MINDIR and MAXDIR are not used.",
    )
    LSTAG: Optional[str] = Field(
        "F",
        description="Boolean flag to stagger directional bins with a half Dtheta. Can only be True for regular grids to avoid characteristic lines aligning with grid lines.",
    )
    MINDIR: Optional[float] = Field(
        0.0,
        description="Minimum direction for simulation in degrees (nautical convention; 0: from N; 90: from E). Not used if LCIRD is True.",
    )
    MAXDIR: Optional[float] = Field(
        360.0,
        description="Maximum direction for simulation in degrees. May be less than MINDIR. Not used if LCIRD is True.",
    )
    MDC: Optional[int] = Field(
        36, description="Number of directional bins for the simulation."
    )
    FRLOW: Optional[float] = Field(
        0.04,
        description="Low frequency limit of the discrete wave period in Hz (1/period).",
    )
    FRHIGH: Optional[float] = Field(
        1.0, description="High frequency limit of the discrete wave period in Hz."
    )
    MSC: Optional[int] = Field(
        36, description="Number of frequency bins for the simulation."
    )
    IGRIDTYPE: Optional[int] = Field(
        3,
        description="Grid type used for the simulation. 1: XFN, 2: WWM-PERIODIC, 3: SCHISM, 4: OLD WWM GRID",
    )
    FILEGRID: Optional[str] = Field(
        "'hgrid_WWM.gr3'",
        description="Name of the grid file. Should be 'hgrid_WWM.gr3' if IGRIDTYPE is 3 (SCHISM).",
    )
    LSLOP: Optional[str] = Field(
        "F", description="Boolean flag to enable bottom slope limiter."
    )
    SLMAX: Optional[float] = Field(
        0.2, description="Maximum slope value when bottom slope limiter is enabled."
    )
    LVAR1D: Optional[str] = Field(
        "F",
        description="Boolean flag for 1D mode with variable dx. Not used with SCHISM.",
    )

    @field_validator("LCIRD")
    @classmethod
    def validate_lcird(cls, v: bool) -> bool:
        return v

    @field_validator("LSTAG")
    @classmethod
    def validate_lstag(cls, v: bool) -> bool:
        return v

    @field_validator("MINDIR")
    @classmethod
    def validate_mindir(cls, v: float) -> float:
        if v < 0 or v >= 360:
            raise ValueError("MINDIR must be between 0 and 360")
        return v

    @field_validator("MAXDIR")
    @classmethod
    def validate_maxdir(cls, v: float) -> float:
        if v <= 0 or v > 360:
            raise ValueError("MAXDIR must be between 0 and 360")
        return v

    @field_validator("MDC")
    @classmethod
    def validate_mdc(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MDC must be a positive integer")
        return v

    @field_validator("FRLOW")
    @classmethod
    def validate_frlow(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("FRLOW must be positive")
        return v

    @field_validator("FRHIGH")
    @classmethod
    def validate_frhigh(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("FRHIGH must be positive")
        return v

    @field_validator("MSC")
    @classmethod
    def validate_msc(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MSC must be a positive integer")
        return v

    @field_validator("IGRIDTYPE")
    @classmethod
    def validate_igridtype(cls, v: int) -> int:
        if v not in [1, 2, 3, 4]:
            raise ValueError("IGRIDTYPE must be 1, 2, 3, or 4")
        return v

    @field_validator("FILEGRID")
    @classmethod
    def validate_filegrid(cls, v: str) -> str:
        if not v:
            raise ValueError("FILEGRID must not be empty")
        return v

    @field_validator("LSLOP")
    @classmethod
    def validate_lslop(cls, v: bool) -> bool:
        return v

    @field_validator("SLMAX")
    @classmethod
    def validate_slmax(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("SLMAX must be positive")
        return v

    @field_validator("LVAR1D")
    @classmethod
    def validate_lvar1d(cls, v: bool) -> bool:
        return v

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
    LBCSE: Optional[str] = Field(
        "F", description="Flag indicating if wave boundary data is time-dependent"
    )
    LBINTER: Optional[str] = Field(
        "F",
        description="Flag to enable time interpolation when LBCSE is True (not available for quasi-steady mode within subtime steps)",
    )
    LBCWA: Optional[str] = Field(
        "T", description="Flag to enable parametric wave spectra"
    )
    LINHOM: Optional[str] = Field(
        "F", description="Flag for non-uniform wave boundary conditions in space"
    )
    LBCSP: Optional[str] = Field(
        "F",
        description="Flag to specify non-parametric wave spectra (defined in FILEWAVE)",
    )
    LINDSPRDEG: Optional[str] = Field(
        "F",
        description="Flag indicating if directional spreading input is in degrees (True) or exponent (False) for 1-D wave spectra",
    )
    LPARMDIR: Optional[str] = Field(
        "F",
        description="Flag to read directional spreading from WBDS in exponential format (only valid for 1D spectra)",
    )
    FILEWAVE: Optional[str] = Field(
        "'wwmbnd.gr3'",
        description="Filename for boundary file containing discrete wave spectra",
    )
    LBSP1D: Optional[str] = Field(
        "F",
        description="Flag for 1D (frequency space only) format in FILEWAVE when LBCSP is True and LINHOM is False",
    )
    LBSP2D: Optional[str] = Field(
        "F",
        description="Flag for 2D (frequency + directional space) format (not functional)",
    )
    BEGTC: Optional[str] = Field(
        "'19980901.000000'",
        description="Begin time of the wave boundary file (FILEWAVE)",
    )
    DELTC: Optional[int] = Field(1, description="Time step in FILEWAVE")
    UNITC: Optional[str] = Field(
        "'HR'", description="Time unit for DELTC (HR, MIN, or SEC)"
    )
    ENDTC: Optional[str] = Field(
        "'19981002.000000'", description="End time of the wave boundary file"
    )
    FILEBOUND: Optional[str] = Field(
        "'wwmbnd.gr3'",
        description="Filename for boundary file defining boundary and Neumann nodes",
    )
    IBOUNDFORMAT: Optional[int] = Field(
        1,
        description="Boundary format indicator (1: WWM, 3: WW3 2D spectra in netCDF, 6: WW3 2D spectra in netCDF with prescribed spectra)",
    )
    WBHS: Optional[float] = Field(
        4.0,
        description="Significant wave height at the boundary for parametric spectra",
    )
    WBSS: Optional[float] = Field(
        2.0, description="Spectral shape parameter for parametric spectra"
    )
    WBTP: Optional[float] = Field(
        8.0, description="Peak or mean wave period at the boundary (seconds)"
    )
    WBDM: Optional[float] = Field(
        90.0, description="Average wave direction at the boundary (degrees)"
    )
    WBDSMS: Optional[float] = Field(
        1.0,
        description="Flag indicating if directional spreading value is in degrees (1) or exponent (2)",
    )
    WBDS: Optional[float] = Field(
        20.0, description="Directional spreading at the boundary (degrees or exponent)"
    )
    WBGAUSS: Optional[float] = Field(
        0.1, description="Factor for Gaussian distribution if WBSS=4"
    )
    WBPKEN: Optional[float] = Field(
        3.3, description="Peak enhancement factor for JONSWAP spectra if WBSS=2"
    )
    NCDF_HS_NAME: Optional[str] = Field(
        "'hs'", description="NetCDF variable name for significant wave height"
    )
    NCDF_DIR_NAME: Optional[str] = Field(
        "'dir'", description="NetCDF variable name for mean wave direction"
    )
    NCDF_SPR_NAME: Optional[str] = Field(
        "'spr'", description="NetCDF variable name for mean directional spreading"
    )
    NCDF_FP_NAME: Optional[str] = Field(
        "'fp'", description="NetCDF variable name for peak frequency"
    )
    NCDF_F02_NAME: Optional[str] = Field(
        "'t02'", description="NetCDF variable name for zero down-crossing frequency"
    )

    @field_validator("LBCSE")
    def validate_lbcse(cls, v):
        return v

    @field_validator("LBINTER")
    def validate_lbinter(cls, v):
        return v

    @field_validator("LBCWA")
    def validate_lbcwa(cls, v):
        return v

    @field_validator("LINHOM")
    def validate_linhom(cls, v):
        return v

    @field_validator("LBCSP")
    def validate_lbcsp(cls, v):
        return v

    @field_validator("LINDSPRDEG")
    def validate_lindsprdeg(cls, v):
        return v

    @field_validator("LPARMDIR")
    def validate_lparmdir(cls, v):
        return v

    @field_validator("FILEWAVE")
    def validate_filewave(cls, v):
        if not v.endswith(".gr3"):
            raise ValueError("FILEWAVE must end with .gr3")
        return v

    @field_validator("LBSP1D")
    def validate_lbsp1d(cls, v):
        return v

    @field_validator("LBSP2D")
    def validate_lbsp2d(cls, v):
        return v

    @field_validator("BEGTC")
    def validate_begtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
        except ValueError:
            raise ValueError("BEGTC must be in format 'YYYYMMDD.HHMMSS'")
        return v

    @field_validator("DELTC")
    def validate_deltc(cls, v):
        if v <= 0:
            raise ValueError("DELTC must be positive")
        return v

    @field_validator("UNITC")
    def validate_unitc(cls, v):
        if v not in ["HR", "MIN", "SEC"]:
            raise ValueError("UNITC must be 'HR', 'MIN', or 'SEC'")
        return v

    @field_validator("ENDTC")
    def validate_endtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
        except ValueError:
            raise ValueError("ENDTC must be in format 'YYYYMMDD.HHMMSS'")
        return v

    @field_validator("FILEBOUND")
    def validate_filebound(cls, v):
        if not v.endswith(".gr3"):
            raise ValueError("FILEBOUND must end with .gr3")
        return v

    @field_validator("IBOUNDFORMAT")
    def validate_iboundformat(cls, v):
        if v not in [1, 3, 6]:
            raise ValueError("IBOUNDFORMAT must be 1, 3, or 6")
        return v

    @field_validator("WBHS")
    def validate_wbhs(cls, v):
        if v <= 0:
            raise ValueError("WBHS must be positive")
        return v

    @field_validator("WBSS")
    def validate_wbss(cls, v):
        if v not in [-3, -2, -1, 1, 2, 3, 4]:
            raise ValueError("WBSS must be -3, -2, -1, 1, 2, 3, or 4")
        return v

    @field_validator("WBTP")
    def validate_wbtp(cls, v):
        if v <= 0:
            raise ValueError("WBTP must be positive")
        return v

    @field_validator("WBDM")
    def validate_wbdm(cls, v):
        if not 0 <= v <= 360:
            raise ValueError("WBDM must be between 0 and 360")
        return v

    @field_validator("WBDSMS")
    def validate_wbdsms(cls, v):
        if v not in [1, 2]:
            raise ValueError("WBDSMS must be 1 or 2")
        return v

    @field_validator("WBDS")
    def validate_wbds(cls, v):
        if v <= 0:
            raise ValueError("WBDS must be positive")
        return v

    @field_validator("WBGAUSS")
    def validate_wbgauss(cls, v):
        if v <= 0:
            raise ValueError("WBGAUSS must be positive")
        return v

    @field_validator("WBPKEN")
    def validate_wbpken(cls, v):
        if v <= 0:
            raise ValueError("WBPKEN must be positive")
        return v

    @field_validator("NCDF_HS_NAME")
    def validate_ncdf_hs_name(cls, v):
        return v

    @field_validator("NCDF_DIR_NAME")
    def validate_ncdf_dir_name(cls, v):
        return v

    @field_validator("NCDF_SPR_NAME")
    def validate_ncdf_spr_name(cls, v):
        return v

    @field_validator("NCDF_FP_NAME")
    def validate_ncdf_fp_name(cls, v):
        return v

    @field_validator("NCDF_F02_NAME")
    def validate_ncdf_f02_name(cls, v):
        return v

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
    MESNL: Optional[int] = Field(
        1,
        description="Controls the nonlinear wave-wave interaction calculation using the Discrete Interaction Approximation (DIA). 1 enables the calculation, 0 disables it.",
    )
    MESIN: Optional[int] = Field(
        1,
        description="Specifies the wind input formulation. Options include: 1 (Ardhuin et al.), 2 (ECMWF physics), 3 (Makin & Stam), 4 (Babanin et al.), 5 (Cycle 3), 0 (no wind).",
    )
    IFRIC: Optional[int] = Field(
        1,
        description="Specifies the formulation for the atmospheric boundary layer. Use 1 when MESIN=1, and 4 when MESIN=3.",
    )
    MESBF: Optional[int] = Field(
        1,
        description="Specifies the bottom friction formulation. 1 for JONSWAP (Default), 2 for Madsen et al. (1989), 3 for SHOWEX.",
    )
    FRICC: Optional[float] = Field(
        0.067,
        description="Bottom friction coefficient. For MESBF=1: JONSWAP coefficient [0.038,0.067]. For MESBF=2: physical bottom roughness. For MESBF=3: D50 (if negative, read from SHOWEX_D50.gr3).",
    )
    MESBR: Optional[int] = Field(
        1,
        description="Controls shallow water wave breaking calculation. 0 disables it, 1 enables it.",
    )
    IBREAK: Optional[int] = Field(
        1,
        description="Specifies the wave breaking formulation. Options range from 1 to 6, each representing a different method.",
    )
    ICRIT: Optional[int] = Field(
        1,
        description="Specifies the wave breaking criterion. Options range from 1 to 6, each representing a different method or calculation.",
    )
    BRCR: Optional[float] = Field(
        0.78,
        description="Breaking criterion parameter. Its meaning depends on IBREAK and ICRIT values.",
    )
    a_BRCR: Optional[float] = Field(
        0.76,
        description="Coefficient used in breaking criterion calculations when ICRIT is 4 or 5.",
    )
    b_BRCR: Optional[float] = Field(
        0.29,
        description="Coefficient used in breaking criterion calculations when ICRIT is 4 or 5.",
    )
    min_BRCR: Optional[float] = Field(
        0.25, description="Minimum value for breaking criterion when ICRIT is 4 or 5."
    )
    max_BRCR: Optional[float] = Field(
        0.8, description="Maximum value for breaking criterion when ICRIT is 4 or 5."
    )
    a_BIPH: Optional[float] = Field(
        0.2, description="Biphase coefficient, used when IBREAK is 3."
    )
    BR_COEF_METHOD: Optional[int] = Field(
        1,
        description="Method for calculating the breaking coefficient. 1 for constant, 2 for adaptive.",
    )
    B_ALP: Optional[float] = Field(
        0.5,
        description="Breaking coefficient. If BR_COEF_METHOD is 2, B_ALP should be around 40.",
    )
    ZPROF_BREAK: Optional[int] = Field(
        2,
        description="Specifies the vertical distribution function of wave breaking source term in 3D runs. Options range from 1 to 6.",
    )
    BC_BREAK: Optional[int] = Field(
        1,
        description="Controls the application of depth-limited breaking at boundaries. 1 enables it, 0 disables it.",
    )
    IROLLER: Optional[int] = Field(
        0,
        description="Controls the wave roller model. 1 enables it, 0 disables it. Currently not used.",
    )
    ALPROL: Optional[float] = Field(
        0.85,
        description="Alpha coefficient for the wave roller model. Range is between 0 and 1.",
    )
    MEVEG: Optional[int] = Field(
        0,
        description="Controls vegetation effects. 1 enables it, 0 disables it. If enabled, isav must be 1 in param.nml.",
    )
    LMAXETOT: Optional[str] = Field(
        "T",
        description="Controls the limitation of shallow water wave height by wave breaking limiter.",
    )
    MESDS: Optional[int] = Field(
        1,
        description="Specifies the formulation for the whitecapping source function. Should have the same value as MESIN.",
    )
    MESTR: Optional[int] = Field(
        1,
        description="Specifies the formulation for the triad 3 wave interactions. 0 disables it, 1 uses Lumped Triad Approx. (LTA), 2 uses corrected version of LTA by Salmon et al. (2016).",
    )
    TRICO: Optional[float] = Field(
        0.1, description="Proportionality constant (alpha_EB) for triad interactions."
    )
    TRIRA: Optional[float] = Field(
        2.5,
        description="Ratio of maximum frequency considered in triads over mean frequency.",
    )
    TRIURS: Optional[float] = Field(
        0.1,
        description="Critical Ursell number for triad calculations. Triads are not computed if Ursell number is less than TRIURS.",
    )

    @field_validator("MESNL")
    @classmethod
    def validate_mesnl(cls, v):
        if v not in [0, 1]:
            raise ValueError("MESNL must be 0 or 1")
        return v

    @field_validator("MESIN")
    @classmethod
    def validate_mesin(cls, v):
        if v not in range(6):
            raise ValueError("MESIN must be between 0 and 5")
        return v

    @field_validator("IFRIC")
    @classmethod
    def validate_ifric(cls, v):
        if v not in [1, 4]:
            raise ValueError("IFRIC must be 1 or 4")
        return v

    @field_validator("MESBF")
    @classmethod
    def validate_mesbf(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("MESBF must be 1, 2, or 3")
        return v

    @field_validator("FRICC")
    @classmethod
    def validate_fricc(cls, v):
        return v

    @field_validator("MESBR")
    @classmethod
    def validate_mesbr(cls, v):
        if v not in [0, 1]:
            raise ValueError("MESBR must be 0 or 1")
        return v

    @field_validator("IBREAK")
    @classmethod
    def validate_ibreak(cls, v):
        if v not in range(1, 7):
            raise ValueError("IBREAK must be between 1 and 6")
        return v

    @field_validator("ICRIT")
    @classmethod
    def validate_icrit(cls, v):
        if v not in range(1, 7):
            raise ValueError("ICRIT must be between 1 and 6")
        return v

    @field_validator("BRCR")
    @classmethod
    def validate_brcr(cls, v):
        return v

    @field_validator("a_BRCR")
    @classmethod
    def validate_a_brcr(cls, v):
        return v

    @field_validator("b_BRCR")
    @classmethod
    def validate_b_brcr(cls, v):
        return v

    @field_validator("min_BRCR")
    @classmethod
    def validate_min_brcr(cls, v):
        return v

    @field_validator("max_BRCR")
    @classmethod
    def validate_max_brcr(cls, v):
        return v

    @field_validator("a_BIPH")
    @classmethod
    def validate_a_biph(cls, v):
        return v

    @field_validator("BR_COEF_METHOD")
    @classmethod
    def validate_br_coef_method(cls, v):
        if v not in [1, 2]:
            raise ValueError("BR_COEF_METHOD must be 1 or 2")
        return v

    @field_validator("B_ALP")
    @classmethod
    def validate_b_alp(cls, v):
        return v

    @field_validator("ZPROF_BREAK")
    @classmethod
    def validate_zprof_break(cls, v):
        if v not in range(1, 7):
            raise ValueError("ZPROF_BREAK must be between 1 and 6")
        return v

    @field_validator("BC_BREAK")
    @classmethod
    def validate_bc_break(cls, v):
        if v not in [0, 1]:
            raise ValueError("BC_BREAK must be 0 or 1")
        return v

    @field_validator("IROLLER")
    @classmethod
    def validate_iroller(cls, v):
        if v not in [0, 1]:
            raise ValueError("IROLLER must be 0 or 1")
        return v

    @field_validator("ALPROL")
    @classmethod
    def validate_alprol(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("ALPROL must be between 0 and 1")
        return v

    @field_validator("MEVEG")
    @classmethod
    def validate_meveg(cls, v):
        if v not in [0, 1]:
            raise ValueError("MEVEG must be 0 or 1")
        return v

    @field_validator("LMAXETOT")
    @classmethod
    def validate_lmaxetot(cls, v):
        if not isinstance(v, bool):
            raise ValueError("LMAXETOT must be a boolean")
        return v

    @field_validator("MESDS")
    @classmethod
    def validate_mesds(cls, v):
        return v

    @field_validator("MESTR")
    @classmethod
    def validate_mestr(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("MESTR must be 0, 1, or 2")
        return v

    @field_validator("TRICO")
    @classmethod
    def validate_trico(cls, v):
        return v

    @field_validator("TRIRA")
    @classmethod
    def validate_trira(cls, v):
        return v

    @field_validator("TRIURS")
    @classmethod
    def validate_triurs(cls, v):
        return v

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
    ZWND: Optional[list] = Field(
        [10.0, ""],
        description="Wind input height in meters. Used to define the reference height for wind measurements.",
    )
    ALPHA0: Optional[list] = Field(
        ["9.499999694526196E-003", ""],
        description="Charnock coefficient for wind stress calculation. Controls the roughness of the sea surface.",
    )
    Z0MAX: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Maximum value for the roughness length. Limits the sea surface roughness in high wind conditions.",
    )
    BETAMAX: Optional[list] = Field(
        [1.54, ""],
        description="Maximum value for the wave growth parameter. Used for tuning the model to specific wind data sources (e.g., ECMWF = 1.52, CFRS = 1.34).",
    )
    SINTHP: Optional[list] = Field(
        [2.0, ""],
        description="Power of cosine in angular distribution of wind input. Controls the directional spread of wind-generated waves.",
    )
    ZALP: Optional[list] = Field(
        ["6.000000052154064E-003", ""],
        description="Wave age parameter for wind input. Affects the growth rate of young wind waves.",
    )
    TAUWSHELTER: Optional[list] = Field(
        [0.300000011920929, ""],
        description="Sheltering coefficient for short waves. Affects the energy transfer from wind to waves in the presence of longer waves.",
    )
    SWELLFPAR: Optional[list] = Field(
        [1.0, ""],
        description="Swell dissipation parameter. Controls the overall strength of swell dissipation.",
    )
    SWELLF: Optional[list] = Field(
        [0.660000026226044, ""],
        description="Swell dissipation coefficient. Part of the swell dissipation parameterization.",
    )
    SWELLF2: Optional[list] = Field(
        ["-1.799999922513962E-002", ""],
        description="Second swell dissipation coefficient. Affects the dependence of swell dissipation on wave steepness.",
    )
    SWELLF3: Optional[list] = Field(
        ["2.199999988079071E-002", ""],
        description="Third swell dissipation coefficient. Modifies the swell dissipation rate.",
    )
    SWELLF4: Optional[list] = Field(
        [150000.0, ""],
        description="Reynolds number threshold for swell dissipation. Controls the onset of turbulent flow around swell.",
    )
    SWELLF5: Optional[list] = Field(
        [1.20000004768372, ""],
        description="Coefficient for swell dissipation in turbulent flow. Affects the strength of swell dissipation above the Reynolds number threshold.",
    )
    SWELLF6: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Lower limit of Reynolds number for swell dissipation. Sets a minimum threshold for swell dissipation.",
    )
    SWELLF7: Optional[list] = Field(
        [360000.0, ""],
        description="Upper limit of Reynolds number for swell dissipation. Sets a maximum threshold for swell dissipation.",
    )
    Z0RAT: Optional[list] = Field(
        ["3.999999910593033E-002", ""],
        description="Ratio of roughness lengths for momentum and energy transfer. Affects the coupling between wind and waves.",
    )
    SINBR: Optional[str] = Field(
        "0.000000000000000E+000",
        description="Bottom friction coefficient for wave dissipation. Controls the rate of energy loss due to bottom friction.",
    )

    @field_validator("ZWND")
    @classmethod
    def validate_zwnd(cls, v):
        if v <= 0:
            raise ValueError("ZWND must be positive")
        return v

    @field_validator("ALPHA0")
    @classmethod
    def validate_alpha0(cls, v):
        if v < 0 or v > 1:
            raise ValueError("ALPHA0 must be between 0 and 1")
        return v

    @field_validator("Z0MAX")
    @classmethod
    def validate_z0max(cls, v):
        if v < 0:
            raise ValueError("Z0MAX must be non-negative")
        return v

    @field_validator("BETAMAX")
    @classmethod
    def validate_betamax(cls, v):
        if v <= 0:
            raise ValueError("BETAMAX must be positive")
        return v

    @field_validator("SINTHP")
    @classmethod
    def validate_sinthp(cls, v):
        if v <= 0:
            raise ValueError("SINTHP must be positive")
        return v

    @field_validator("ZALP")
    @classmethod
    def validate_zalp(cls, v):
        if v < 0 or v > 1:
            raise ValueError("ZALP must be between 0 and 1")
        return v

    @field_validator("TAUWSHELTER")
    @classmethod
    def validate_tauwshelter(cls, v):
        if v < 0 or v > 1:
            raise ValueError("TAUWSHELTER must be between 0 and 1")
        return v

    @field_validator("SWELLFPAR")
    @classmethod
    def validate_swellfpar(cls, v):
        if v <= 0:
            raise ValueError("SWELLFPAR must be positive")
        return v

    @field_validator("SWELLF")
    @classmethod
    def validate_swellf(cls, v):
        if v < 0 or v > 1:
            raise ValueError("SWELLF must be between 0 and 1")
        return v

    @field_validator("SWELLF2")
    @classmethod
    def validate_swellf2(cls, v):
        return v

    @field_validator("SWELLF3")
    @classmethod
    def validate_swellf3(cls, v):
        return v

    @field_validator("SWELLF4")
    @classmethod
    def validate_swellf4(cls, v):
        if v <= 0:
            raise ValueError("SWELLF4 must be positive")
        return v

    @field_validator("SWELLF5")
    @classmethod
    def validate_swellf5(cls, v):
        if v <= 0:
            raise ValueError("SWELLF5 must be positive")
        return v

    @field_validator("SWELLF6")
    @classmethod
    def validate_swellf6(cls, v):
        if v < 0:
            raise ValueError("SWELLF6 must be non-negative")
        return v

    @field_validator("SWELLF7")
    @classmethod
    def validate_swellf7(cls, v):
        if v <= 0:
            raise ValueError("SWELLF7 must be positive")
        return v

    @field_validator("Z0RAT")
    @classmethod
    def validate_z0rat(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("Z0RAT must be between 0 and 1")
        return v

    @field_validator("SINBR")
    @classmethod
    def validate_sinbr(cls, v):
        if v < 0:
            raise ValueError("SINBR must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_swellf_limits(self):
        if self.SWELLF6 >= self.SWELLF7:
            raise ValueError("SWELLF6 must be less than SWELLF7")
        return self


class Sds4(NamelistBaseModel):
    SDSC1: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Constant parameter in source term formulation",
    )
    FXPM3: Optional[list] = Field(
        [4.0, ""], description="Scaling factor for wave steepness in source term"
    )
    FXFM3: Optional[list] = Field(
        [2.5, ""], description="Scaling factor for wave frequency in source term"
    )
    FXFMAGE: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Age-dependent factor in source term formulation",
    )
    SDSC2: Optional[list] = Field(
        ["-2.200000017182902E-005", ""],
        description="Second constant parameter in source term formulation",
    )
    SDSCUM: Optional[list] = Field(
        [-0.403439998626709, ""], description="Cumulative source term parameter"
    )
    SDSSTRAIN: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Strain-related parameter in source term",
    )
    SDSC4: Optional[list] = Field(
        [1.0, ""], description="Fourth constant parameter in source term formulation"
    )
    SDSC5: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Fifth constant parameter in source term formulation",
    )
    SDSC6: Optional[list] = Field(
        [0.300000011920929, ""],
        description="Sixth constant parameter in source term formulation",
    )
    SDSBR: Optional[list] = Field(
        ["8.999999845400453E-004", ""],
        description="Breaking-related parameter in source term",
    )
    SDSBR2: Optional[list] = Field(
        [0.800000011920929, ""], description="Secondary breaking-related parameter"
    )
    SDSP: Optional[list] = Field(
        [2.0, ""], description="Power-related parameter in source term"
    )
    SDSISO: Optional[list] = Field(
        [2.0, ""], description="Isotropic dissipation parameter"
    )
    SDSBCK: Optional[list] = Field(
        ["0.000000000000000E+000", ""], description="Background dissipation parameter"
    )
    SDSABK: Optional[list] = Field(
        [1.5, ""], description="Additional background dissipation parameter"
    )
    SDSPBK: Optional[list] = Field(
        [4.0, ""], description="Power-related background dissipation parameter"
    )
    SDSBINT: Optional[list] = Field(
        [0.300000011920929, ""], description="Intermediate-scale dissipation parameter"
    )
    SDSHCK: Optional[list] = Field(
        [1.5, ""], description="High-frequency check parameter"
    )
    SDSDTH: Optional[list] = Field(
        [80.0, ""], description="Directional spreading threshold parameter"
    )
    SDSCOS: Optional[list] = Field(
        [2.0, ""], description="Cosine power parameter in directional spreading"
    )
    SDSBRF1: Optional[list] = Field(
        [0.5, ""], description="Breaking-related frequency parameter"
    )
    SDSBRFDF: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Breaking-related frequency difference parameter",
    )
    SDSBM0: Optional[list] = Field(
        [1.0, ""], description="Breaking-related moment parameter 0"
    )
    SDSBM1: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Breaking-related moment parameter 1",
    )
    SDSBM2: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Breaking-related moment parameter 2",
    )
    SDSBM3: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Breaking-related moment parameter 3",
    )
    SDSBM4: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="Breaking-related moment parameter 4",
    )
    SDSHFGEN: Optional[list] = Field(
        ["0.000000000000000E+000", ""],
        description="High-frequency generation parameter",
    )
    SDSLFGEN: Optional[list] = Field(
        ["0.000000000000000E+000", ""], description="Low-frequency generation parameter"
    )
    WHITECAPWIDTH: Optional[list] = Field(
        [0.300000011920929, ""], description="Width of the whitecap region"
    )
    FXINCUT: Optional[list] = Field(
        ["0.000000000000000E+000", ""], description="Input cut-off frequency parameter"
    )
    FXDSCUT: Optional[str] = Field(
        "0.000000000000000E+000", description="Dissipation cut-off frequency parameter"
    )

    @field_validator("SDSC1")
    @classmethod
    def validate_sdsc1(cls, v):
        return float(v)

    @field_validator("FXPM3")
    @classmethod
    def validate_fxpm3(cls, v):
        if v <= 0:
            raise ValueError("FXPM3 must be positive")
        return float(v)

    @field_validator("FXFM3")
    @classmethod
    def validate_fxfm3(cls, v):
        if v <= 0:
            raise ValueError("FXFM3 must be positive")
        return float(v)

    @field_validator("FXFMAGE")
    @classmethod
    def validate_fxfmage(cls, v):
        return float(v)

    @field_validator("SDSC2")
    @classmethod
    def validate_sdsc2(cls, v):
        return float(v)

    @field_validator("SDSCUM")
    @classmethod
    def validate_sdscum(cls, v):
        return float(v)

    @field_validator("SDSSTRAIN")
    @classmethod
    def validate_sdsstrain(cls, v):
        return float(v)

    @field_validator("SDSC4")
    @classmethod
    def validate_sdsc4(cls, v):
        return float(v)

    @field_validator("SDSC5")
    @classmethod
    def validate_sdsc5(cls, v):
        return float(v)

    @field_validator("SDSC6")
    @classmethod
    def validate_sdsc6(cls, v):
        return float(v)

    @field_validator("SDSBR")
    @classmethod
    def validate_sdsbr(cls, v):
        if v < 0:
            raise ValueError("SDSBR must be non-negative")
        return float(v)

    @field_validator("SDSBR2")
    @classmethod
    def validate_sdsbr2(cls, v):
        if v < 0 or v > 1:
            raise ValueError("SDSBR2 must be between 0 and 1")
        return float(v)

    @field_validator("SDSP")
    @classmethod
    def validate_sdsp(cls, v):
        if v <= 0:
            raise ValueError("SDSP must be positive")
        return float(v)

    @field_validator("SDSISO")
    @classmethod
    def validate_sdsiso(cls, v):
        if v <= 0:
            raise ValueError("SDSISO must be positive")
        return float(v)

    @field_validator("SDSBCK")
    @classmethod
    def validate_sdsbck(cls, v):
        return float(v)

    @field_validator("SDSABK")
    @classmethod
    def validate_sdsabk(cls, v):
        if v < 0:
            raise ValueError("SDSABK must be non-negative")
        return float(v)

    @field_validator("SDSPBK")
    @classmethod
    def validate_sdspbk(cls, v):
        if v <= 0:
            raise ValueError("SDSPBK must be positive")
        return float(v)

    @field_validator("SDSBINT")
    @classmethod
    def validate_sdsbint(cls, v):
        if v < 0 or v > 1:
            raise ValueError("SDSBINT must be between 0 and 1")
        return float(v)

    @field_validator("SDSHCK")
    @classmethod
    def validate_sdshck(cls, v):
        if v <= 0:
            raise ValueError("SDSHCK must be positive")
        return float(v)

    @field_validator("SDSDTH")
    @classmethod
    def validate_sdsdth(cls, v):
        if v <= 0 or v > 360:
            raise ValueError("SDSDTH must be between 0 and 360")
        return float(v)

    @field_validator("SDSCOS")
    @classmethod
    def validate_sdscos(cls, v):
        if v <= 0:
            raise ValueError("SDSCOS must be positive")
        return float(v)

    @field_validator("SDSBRF1")
    @classmethod
    def validate_sdsbrf1(cls, v):
        if v < 0 or v > 1:
            raise ValueError("SDSBRF1 must be between 0 and 1")
        return float(v)

    @field_validator("SDSBRFDF")
    @classmethod
    def validate_sdsbrfdf(cls, v):
        return float(v)

    @field_validator("SDSBM0")
    @classmethod
    def validate_sdsbm0(cls, v):
        return float(v)

    @field_validator("SDSBM1")
    @classmethod
    def validate_sdsbm1(cls, v):
        return float(v)

    @field_validator("SDSBM2")
    @classmethod
    def validate_sdsbm2(cls, v):
        return float(v)

    @field_validator("SDSBM3")
    @classmethod
    def validate_sdsbm3(cls, v):
        return float(v)

    @field_validator("SDSBM4")
    @classmethod
    def validate_sdsbm4(cls, v):
        return float(v)

    @field_validator("SDSHFGEN")
    @classmethod
    def validate_sdshfgen(cls, v):
        return float(v)

    @field_validator("SDSLFGEN")
    @classmethod
    def validate_sdslfgen(cls, v):
        return float(v)

    @field_validator("WHITECAPWIDTH")
    @classmethod
    def validate_whitecapwidth(cls, v):
        if v < 0 or v > 1:
            raise ValueError("WHITECAPWIDTH must be between 0 and 1")
        return float(v)

    @field_validator("FXINCUT")
    @classmethod
    def validate_fxincut(cls, v):
        return float(v)

    @field_validator("FXDSCUT")
    @classmethod
    def validate_fxdscut(cls, v):
        return float(v)


class Nums(NamelistBaseModel):
    ICOMP: Optional[int] = Field(
        3,
        description="Controls the splitting method and implicit/explicit schemes for spectral advection. 0: Explicit for all dimensions, 1: Implicit for geographical space, explicit for others, 2: Implicit for advection and semi-implicit for source terms, 3: Fully implicit with no splitting.",
    )
    AMETHOD: Optional[int] = Field(
        7,
        description="Controls the methods used in geographical space. Values range from 0 to 7, with different schemes and solvers for each value.",
    )
    ASPAR_LOCAL_LEVEL: Optional[list] = Field(
        [0, ""],
        description="Locality level for memory usage. 0 uses a lot of memory, 10 uses no memory, values in between are hybrid levels.",
    )
    SMETHOD: Optional[int] = Field(
        1,
        description="Controls the integration of source terms. 0: No source terms, 1: Splitting using RK-3 and SI, 2: Semi-implicit, 3: R-K3, 4: Dynamic Splitting, 6: Sub-time steps for breaking term integration.",
    )
    DMETHOD: Optional[int] = Field(
        2,
        description="Controls the numerical method in directional space. 0: No advection, 1: Crank-Nicholson or Euler Implicit, 2: Ultimate Quickest, 3: RK5-WENO, 4: Explicit FVM Upwind scheme.",
    )
    RTHETA: Optional[float] = Field(
        0.5,
        description="Weighing factor for DMETHOD = 1. Useful only for Crank Nicholson integration.",
    )
    LITERSPLIT: Optional[str] = Field(
        "F",
        description="Controls splitting method. True: double Strang split, False: simple split (more efficient).",
    )
    LFILTERTH: Optional[str] = Field(
        "F",
        description="Use a CFL filter to limit the advection velocity in directional space.",
    )
    MAXCFLTH: Optional[float] = Field(
        1.0, description="Maximum CFL in Theta space, used only if LFILTERTH=True."
    )
    FMETHOD: Optional[int] = Field(
        1,
        description="Controls the numerical method used in frequency space. 0: No advection, 1: Ultimate Quickest as in WW3.",
    )
    LFILTERSIG: Optional[str] = Field(
        "F", description="Limit the advection velocity in frequency space."
    )
    MAXCFLSIG: Optional[float] = Field(
        1.0, description="Maximum CFL in frequency space, used only if LFILTERSIG=True."
    )
    LLIMT: Optional[str] = Field(
        "T", description="Switch on/off Action limiter. Must mostly be turned on."
    )
    LSIGBOUND: Optional[str] = Field(
        "F", description="Theta space on wet land/island boundary."
    )
    LTHBOUND: Optional[str] = Field(
        "F", description="Sigma space on wet land/island boundary."
    )
    LSOUBOUND: Optional[str] = Field(
        "F",
        description="Source Terms on wet land/island boundary. Use True if SMETHOD=6.",
    )
    MELIM: Optional[int] = Field(
        1,
        description="Formulation for the action limiter. 1: WAM group (1988), 2: Hersbach Janssen (1999), 3: For Cycle 4 formulation.",
    )
    LIMFAK: Optional[float] = Field(
        0.1,
        description="Proportionality coefficient for the action limiter. Value depends on MESIN and MESDS settings.",
    )
    LDIFR: Optional[str] = Field(
        "F",
        description="Use phase decoupled diffraction approximation according to Holthuijsen et al. (2003).",
    )
    IDIFFR: Optional[int] = Field(
        1,
        description="Extended WAE accounting for higher order effects. 1: Holthuijsen et al., 2: Liau et al., 3: Toledo et al.",
    )
    LCONV: Optional[str] = Field(
        "F",
        description="Estimate convergence criteria and write to disk (quasi-steady - qstea.out).",
    )
    LCFL: Optional[str] = Field(
        "F", description="Write out CFL numbers. Use False to save time."
    )
    NQSITER: Optional[int] = Field(
        10,
        description="Number of quasi-steady (Q-S) sub-divisions within each WWM time step.",
    )
    QSCONV1: Optional[float] = Field(
        0.98,
        description="Percentage of grid points that must fulfill absolute wave height criteria EPSH1.",
    )
    QSCONV2: Optional[float] = Field(
        0.98,
        description="Percentage of grid points that must fulfill relative wave height criteria EPSH2.",
    )
    QSCONV3: Optional[float] = Field(
        0.98,
        description="Percentage of grid points that must fulfill sum relative wave action criteria EPSH3.",
    )
    QSCONV4: Optional[float] = Field(
        0.98,
        description="Percentage of grid points that must fulfill average relative wave period criteria EPSH4.",
    )
    QSCONV5: Optional[float] = Field(
        0.98,
        description="Percentage of grid points that must fulfill average relative wave steepness criteria EPSH5.",
    )
    LEXPIMP: Optional[str] = Field(
        "F",
        description="Use implicit schemes for frequencies lower than FREQEXP. Used only if ICOMP=0.",
    )
    FREQEXP: Optional[float] = Field(
        0.1,
        description="Minimum frequency for explicit schemes. Only used if LEXPIMP=True and ICOMP=0.",
    )
    EPSH1: Optional[float] = Field(
        0.01, description="Convergence criteria for relative wave height."
    )
    EPSH2: Optional[float] = Field(
        0.01, description="Convergence criteria for absolute wave height."
    )
    EPSH3: Optional[float] = Field(
        0.01, description="Convergence criteria for the relative sum of wave action."
    )
    EPSH4: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the relative average wave steepness.",
    )
    EPSH5: Optional[float] = Field(
        0.01, description="Convergence criteria for the relative average wave period."
    )
    LVECTOR: Optional[str] = Field(
        "F",
        description="Use optimized propagation routines for large high performance computers.",
    )
    IVECTOR: Optional[int] = Field(
        2,
        description="Different flavors of communications when LVECTOR=True. Values range from 1 to 6.",
    )
    LADVTEST: Optional[str] = Field(
        "F", description="For testing the advection schemes."
    )
    LCHKCONV: Optional[str] = Field(
        "T",
        description="Needs to be set to True for quasi-steady mode to compute and check the QSCONVi criteria.",
    )
    NB_BLOCK: Optional[list] = Field(
        [3, ""], description="Number of blocks for some computational method."
    )
    WAE_SOLVERTHR: Optional[list] = Field(
        ["1.E-6", ""], description="Solver threshold for WAE (Wave Action Equation)."
    )
    MAXITER: Optional[list] = Field(
        [1000, ""],
        description="Maximum number of iterations for some computational method.",
    )
    LSOURCESWAM: Optional[list] = Field(
        ["F", ""], description="Use ECMWF WAM formulation for deep water physics."
    )
    LNANINFCHK: Optional[list] = Field(
        ["F", ""], description="Check for NaN and Inf values."
    )
    LZETA_SETUP: Optional[list] = Field(
        ["F", ""], description="Enable zeta setup calculation."
    )
    ZETA_METH: Optional[list] = Field(
        [0, ""], description="Method for zeta calculation."
    )
    PMIN: Optional[list] = Field(
        [5.0, ""], description="Minimum value for some parameter, possibly pressure."
    )
    BLOCK_GAUSS_SEIDEL: Optional[list] = Field(
        ["T", ""], description="Use Block Gauss-Seidel method."
    )
    LNONL: Optional[list] = Field(
        ["F", ""], description="Enable non-linear calculations."
    )
    L_SOLVER_NORM: Optional[list] = Field(
        ["F", ""], description="Use solver normalization."
    )

    @field_validator("ICOMP")
    @classmethod
    def validate_icomp(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("ICOMP must be 0, 1, 2, or 3")
        return v

    @field_validator("AMETHOD")
    @classmethod
    def validate_amethod(cls, v):
        if v not in range(8):
            raise ValueError("AMETHOD must be between 0 and 7")
        return v

    @field_validator("ASPAR_LOCAL_LEVEL")
    @classmethod
    def validate_aspar_local_level(cls, v):
        if not 0 <= v <= 10:
            raise ValueError("ASPAR_LOCAL_LEVEL must be between 0 and 10")
        return v

    @field_validator("SMETHOD")
    @classmethod
    def validate_smethod(cls, v):
        if v not in [0, 1, 2, 3, 4, 6]:
            raise ValueError("SMETHOD must be 0, 1, 2, 3, 4, or 6")
        return v

    @field_validator("DMETHOD")
    @classmethod
    def validate_dmethod(cls, v):
        if v not in range(5):
            raise ValueError("DMETHOD must be between 0 and 4")
        return v

    @field_validator("RTHETA")
    @classmethod
    def validate_rtheta(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("RTHETA must be between 0 and 1")
        return v

    @field_validator("LITERSPLIT")
    @classmethod
    def validate_litersplit(cls, v):
        return bool(v)

    @field_validator("LFILTERTH")
    @classmethod
    def validate_lfilterth(cls, v):
        return bool(v)

    @field_validator("MAXCFLTH")
    @classmethod
    def validate_maxcflth(cls, v):
        if v <= 0:
            raise ValueError("MAXCFLTH must be positive")
        return v

    @field_validator("FMETHOD")
    @classmethod
    def validate_fmethod(cls, v):
        if v not in [0, 1]:
            raise ValueError("FMETHOD must be 0 or 1")
        return v

    @field_validator("LFILTERSIG")
    @classmethod
    def validate_lfiltersig(cls, v):
        return bool(v)

    @field_validator("MAXCFLSIG")
    @classmethod
    def validate_maxcflsig(cls, v):
        if v <= 0:
            raise ValueError("MAXCFLSIG must be positive")
        return v

    @field_validator("LLIMT")
    @classmethod
    def validate_llimt(cls, v):
        return bool(v)

    @field_validator("LSIGBOUND")
    @classmethod
    def validate_lsigbound(cls, v):
        return bool(v)

    @field_validator("LTHBOUND")
    @classmethod
    def validate_lthbound(cls, v):
        return bool(v)

    @field_validator("LSOUBOUND")
    @classmethod
    def validate_lsoubound(cls, v):
        return bool(v)

    @field_validator("MELIM")
    @classmethod
    def validate_melim(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("MELIM must be 1, 2, or 3")
        return v

    @field_validator("LIMFAK")
    @classmethod
    def validate_limfak(cls, v):
        if v <= 0:
            raise ValueError("LIMFAK must be positive")
        return v

    @field_validator("LDIFR")
    @classmethod
    def validate_ldifr(cls, v):
        return bool(v)

    @field_validator("IDIFFR")
    @classmethod
    def validate_idiffr(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("IDIFFR must be 1, 2, or 3")
        return v

    @field_validator("LCONV")
    @classmethod
    def validate_lconv(cls, v):
        return bool(v)

    @field_validator("LCFL")
    @classmethod
    def validate_lcfl(cls, v):
        return bool(v)

    @field_validator("NQSITER")
    @classmethod
    def validate_nqsiter(cls, v):
        if v <= 0:
            raise ValueError("NQSITER must be positive")
        return v

    @field_validator("QSCONV1")
    @classmethod
    def validate_qsconv1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV1 must be between 0 and 1")
        return v

    @field_validator("QSCONV2")
    @classmethod
    def validate_qsconv2(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV2 must be between 0 and 1")
        return v

    @field_validator("QSCONV3")
    @classmethod
    def validate_qsconv3(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV3 must be between 0 and 1")
        return v

    @field_validator("QSCONV4")
    @classmethod
    def validate_qsconv4(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV4 must be between 0 and 1")
        return v

    @field_validator("QSCONV5")
    @classmethod
    def validate_qsconv5(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV5 must be between 0 and 1")
        return v

    @field_validator("LEXPIMP")
    @classmethod
    def validate_lexpimp(cls, v):
        return bool(v)

    @field_validator("FREQEXP")
    @classmethod
    def validate_freqexp(cls, v):
        if v <= 0:
            raise ValueError("FREQEXP must be positive")
        return v

    @field_validator("EPSH1")
    @classmethod
    def validate_epsh1(cls, v):
        if v <= 0:
            raise ValueError("EPSH1 must be positive")
        return v

    @field_validator("EPSH2")
    @classmethod
    def validate_epsh2(cls, v):
        if v <= 0:
            raise ValueError("EPSH2 must be positive")
        return v

    @field_validator("EPSH3")
    @classmethod
    def validate_epsh3(cls, v):
        if v <= 0:
            raise ValueError("EPSH3 must be positive")
        return v

    @field_validator("EPSH4")
    @classmethod
    def validate_epsh4(cls, v):
        if v <= 0:
            raise ValueError("EPSH4 must be positive")
        return v

    @field_validator("EPSH5")
    @classmethod
    def validate_epsh5(cls, v):
        if v <= 0:
            raise ValueError("EPSH5 must be positive")
        return v

    @field_validator("LVECTOR")
    @classmethod
    def validate_lvector(cls, v):
        return bool(v)

    @field_validator("IVECTOR")
    @classmethod
    def validate_ivector(cls, v):
        if v not in range(1, 7):
            raise ValueError("IVECTOR must be between 1 and 6")
        return v

    @field_validator("LADVTEST")
    @classmethod
    def validate_ladvtest(cls, v):
        return bool(v)

    @field_validator("LCHKCONV")
    @classmethod
    def validate_lchkconv(cls, v):
        return bool(v)

    @field_validator("NB_BLOCK")
    @classmethod
    def validate_nb_block(cls, v):
        if v <= 0:
            raise ValueError("NB_BLOCK must be positive")
        return v

    @field_validator("WAE_SOLVERTHR")
    @classmethod
    def validate_wae_solverthr(cls, v):
        if v <= 0:
            raise ValueError("WAE_SOLVERTHR must be positive")
        return v

    @field_validator("MAXITER")
    @classmethod
    def validate_maxiter(cls, v):
        if v <= 0:
            raise ValueError("MAXITER must be positive")
        return v

    @field_validator("LSOURCESWAM")
    @classmethod
    def validate_lsourceswam(cls, v):
        return bool(v)

    @field_validator("LNANINFCHK")
    @classmethod
    def validate_lnaninfchk(cls, v):
        return bool(v)

    @field_validator("LZETA_SETUP")
    @classmethod
    def validate_lzeta_setup(cls, v):
        return bool(v)

    @field_validator("ZETA_METH")
    @classmethod
    def validate_zeta_meth(cls, v):
        if v not in [0, 1]:
            raise ValueError("ZETA_METH must be 0 or 1")
        return v

    @field_validator("PMIN")
    @classmethod
    def validate_pmin(cls, v):
        if v <= 0:
            raise ValueError("PMIN must be positive")
        return v

    @field_validator("BLOCK_GAUSS_SEIDEL")
    @classmethod
    def validate_block_gauss_seidel(cls, v):
        return bool(v)

    @field_validator("LNONL")
    @classmethod
    def validate_lnonl(cls, v):
        return bool(v)

    @field_validator("L_SOLVER_NORM")
    @classmethod
    def validate_l_solver_norm(cls, v):
        return bool(v)

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
    BEGTC: Optional[str] = Field(
        "'19980901.000000'",
        description="Start output time in 'yyyymmdd.hhmmss' format. Must fit within the simulation time range, otherwise no output is generated. If not specified, defaults to PROC%BEGTC.",
    )
    DELTC: Optional[int] = Field(
        1,
        description="Time step for output in seconds. If smaller than the simulation time step, the latter is used. Useful for better 1D and 2D spectra analysis when set to output every step.",
    )
    UNITC: Optional[str] = Field(
        "'SEC'",
        description="Unit of time for DELTC. Currently only 'SEC' (seconds) is supported.",
    )
    ENDTC: Optional[str] = Field(
        "'19980910.000000'",
        description="Stop time for output in 'yyyymmdd.hhmmss' format. If not specified, defaults to PROC%ENDC.",
    )
    DEFINETC: Optional[int] = Field(
        86400,
        description="Time interval for defining history files in seconds. If unset or negative, only one file is generated. For example, 86400 creates daily output files.",
    )
    OUTSTYLE: Optional[str] = Field(
        "'NO'",
        description="Output option: 'NO' for no output, 'NC' for netCDF, 'XFN' for XFN output, 'SHP' for DARKO SHP output.",
    )
    MULTIPLEOUT: Optional[int] = Field(
        0,
        description="Output style: 0 for single netCDF with MPI_GATHER, 1 for separate netCDF files per process, 2 for parallel netCDF library (not implemented).",
    )
    USE_SINGLE_OUT: Optional[str] = Field(
        "T",
        description="Use single precision in output of model variables. Only impacts if rkind=8 is selected.",
    )
    PARAMWRITE: Optional[str] = Field(
        "T",
        description="Write physical parameterization and chosen numerical discretization in the netCDF history file.",
    )
    GRIDWRITE: Optional[str] = Field(
        "T", description="Write the grid in the netCDF history file."
    )
    PRINTMMA: Optional[str] = Field(
        "F",
        description="Print minimum, maximum, and average value of statistics during runtime.",
    )
    FILEOUT: Optional[str] = Field(
        "'history.dat'", description="Filename for output data."
    )
    LWXFN: Optional[str] = Field("T", description="Enable or disable LWXFN output.")
    HS: Optional[str] = Field("F", description="Output significant wave height.")
    TM01: Optional[str] = Field("F", description="Output mean period.")
    TM02: Optional[str] = Field("F", description="Output zero-crossing mean period.")
    KLM: Optional[str] = Field("F", description="Output mean wave number.")
    WLM: Optional[str] = Field("F", description="Output mean wave length.")
    ETOTC: Optional[str] = Field("F", description="Output variable ETOTC.")
    ETOTS: Optional[str] = Field("F", description="Output variable ETOTS.")
    DM: Optional[str] = Field("T", description="Output mean wave direction.")
    DSPR: Optional[str] = Field("F", description="Output directional spreading.")
    TPPD: Optional[str] = Field("F", description="Output TPPD.")
    TPP: Optional[str] = Field("F", description="Output TPP.")
    CPP: Optional[str] = Field("F", description="Output CPP.")
    WNPP: Optional[str] = Field("F", description="Output peak wave number.")
    CGPP: Optional[str] = Field("F", description="Output peak group speed.")
    KPP: Optional[str] = Field("F", description="Output peak wave number.")
    LPP: Optional[str] = Field("F", description="Output peak wavelength.")
    PEAKD: Optional[str] = Field("F", description="Output peak direction.")
    PEAKDSPR: Optional[str] = Field(
        "F", description="Output peak directional spreading."
    )
    DPEAK: Optional[str] = Field("F", description="Output DPEAK.")
    UBOT: Optional[str] = Field("F", description="Output UBOT.")
    ORBITAL: Optional[str] = Field("F", description="Output ORBITAL.")
    BOTEXPER: Optional[str] = Field("F", description="Output BOTEXPER.")
    TMBOT: Optional[str] = Field("F", description="Output TMBOT.")
    URSELL: Optional[str] = Field("F", description="Output Ursell number.")
    UFRIC: Optional[str] = Field("F", description="Output air friction velocity.")
    Z0: Optional[str] = Field("F", description="Output air roughness length.")
    ALPHA_CH: Optional[str] = Field(
        "F", description="Output Charnoch coefficient for air."
    )
    WINDX: Optional[str] = Field("F", description="Output wind in X direction.")
    WINDY: Optional[str] = Field("F", description="Output wind in Y direction.")
    CD: Optional[str] = Field("F", description="Output drag coefficient.")
    CURRTX: Optional[str] = Field("F", description="Output current in X direction.")
    CURRTY: Optional[str] = Field("F", description="Output current in Y direction.")
    WATLEV: Optional[str] = Field("F", description="Output water level.")
    WATLEVOLD: Optional[str] = Field(
        "F", description="Output water level at previous time step."
    )
    DEP: Optional[str] = Field("F", description="Output depth.")
    TAUW: Optional[str] = Field("F", description="Output surface stress from the wave.")
    TAUHF: Optional[str] = Field(
        "F", description="Output high frequency surface stress."
    )
    TAUTOT: Optional[str] = Field("F", description="Output total surface stress.")
    STOKESSURFX: Optional[str] = Field(
        "F", description="Output surface Stokes drift in X direction."
    )
    STOKESSURFY: Optional[str] = Field(
        "F", description="Output surface Stokes drift in Y direction."
    )
    STOKESBAROX: Optional[str] = Field(
        "F", description="Output barotropic Stokes drift in X direction."
    )
    STOKESBAROY: Optional[str] = Field(
        "F", description="Output barotropic Stokes drift in Y direction."
    )

    @field_validator("BEGTC")
    @classmethod
    def validate_begtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
            return v
        except ValueError:
            raise ValueError("BEGTC must be in format yyyymmdd.hhmmss")

    @field_validator("DELTC")
    @classmethod
    def validate_deltc(cls, v):
        if v <= 0:
            raise ValueError("DELTC must be a positive integer")
        return v

    @field_validator("UNITC")
    @classmethod
    def validate_unitc(cls, v):
        if v != "SEC":
            raise ValueError("UNITC must be SEC")
        return v

    @field_validator("ENDTC")
    @classmethod
    def validate_endtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
            return v
        except ValueError:
            raise ValueError("ENDTC must be in format yyyymmdd.hhmmss")

    @field_validator("DEFINETC")
    @classmethod
    def validate_definetc(cls, v):
        if v < 0:
            return v
        if not isinstance(v, int):
            raise ValueError("DEFINETC must be an integer")
        return v

    @field_validator("OUTSTYLE")
    @classmethod
    def validate_outstyle(cls, v):
        valid_options = ["NO", "NC", "XFN", "SHP"]
        if v not in valid_options:
            raise ValueError(f"OUTSTYLE must be one of {valid_options}")
        return v

    @field_validator("MULTIPLEOUT")
    @classmethod
    def validate_multipleout(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("MULTIPLEOUT must be 0, 1, or 2")
        return v

    @field_validator("USE_SINGLE_OUT")
    @classmethod
    def validate_use_single_out(cls, v):
        return v

    @field_validator("PARAMWRITE")
    @classmethod
    def validate_paramwrite(cls, v):
        return v

    @field_validator("GRIDWRITE")
    @classmethod
    def validate_gridwrite(cls, v):
        return v

    @field_validator("PRINTMMA")
    @classmethod
    def validate_printmma(cls, v):
        return v

    @field_validator("FILEOUT")
    @classmethod
    def validate_fileout(cls, v):
        if not v:
            raise ValueError("FILEOUT must not be empty")
        return v

    @field_validator("LWXFN")
    @classmethod
    def validate_lwxfn(cls, v):
        return v

    @field_validator("HS")
    @classmethod
    def validate_hs(cls, v):
        return v

    @field_validator("TM01")
    @classmethod
    def validate_tm01(cls, v):
        return v

    @field_validator("TM02")
    @classmethod
    def validate_tm02(cls, v):
        return v

    @field_validator("KLM")
    @classmethod
    def validate_klm(cls, v):
        return v

    @field_validator("WLM")
    @classmethod
    def validate_wlm(cls, v):
        return v

    @field_validator("ETOTC")
    @classmethod
    def validate_etotc(cls, v):
        return v

    @field_validator("ETOTS")
    @classmethod
    def validate_etots(cls, v):
        return v

    @field_validator("DM")
    @classmethod
    def validate_dm(cls, v):
        return v

    @field_validator("DSPR")
    @classmethod
    def validate_dspr(cls, v):
        return v

    @field_validator("TPPD")
    @classmethod
    def validate_tppd(cls, v):
        return v

    @field_validator("TPP")
    @classmethod
    def validate_tpp(cls, v):
        return v

    @field_validator("CPP")
    @classmethod
    def validate_cpp(cls, v):
        return v

    @field_validator("WNPP")
    @classmethod
    def validate_wnpp(cls, v):
        return v

    @field_validator("CGPP")
    @classmethod
    def validate_cgpp(cls, v):
        return v

    @field_validator("KPP")
    @classmethod
    def validate_kpp(cls, v):
        return v

    @field_validator("LPP")
    @classmethod
    def validate_lpp(cls, v):
        return v

    @field_validator("PEAKD")
    @classmethod
    def validate_peakd(cls, v):
        return v

    @field_validator("PEAKDSPR")
    @classmethod
    def validate_peakdspr(cls, v):
        return v

    @field_validator("DPEAK")
    @classmethod
    def validate_dpeak(cls, v):
        return v

    @field_validator("UBOT")
    @classmethod
    def validate_ubot(cls, v):
        return v

    @field_validator("ORBITAL")
    @classmethod
    def validate_orbital(cls, v):
        return v

    @field_validator("BOTEXPER")
    @classmethod
    def validate_botexper(cls, v):
        return v

    @field_validator("TMBOT")
    @classmethod
    def validate_tmbot(cls, v):
        return v

    @field_validator("URSELL")
    @classmethod
    def validate_ursell(cls, v):
        return v

    @field_validator("UFRIC")
    @classmethod
    def validate_ufric(cls, v):
        return v

    @field_validator("Z0")
    @classmethod
    def validate_z0(cls, v):
        return v

    @field_validator("ALPHA_CH")
    @classmethod
    def validate_alpha_ch(cls, v):
        return v

    @field_validator("WINDX")
    @classmethod
    def validate_windx(cls, v):
        return v

    @field_validator("WINDY")
    @classmethod
    def validate_windy(cls, v):
        return v

    @field_validator("CD")
    @classmethod
    def validate_cd(cls, v):
        return v

    @field_validator("CURRTX")
    @classmethod
    def validate_currtx(cls, v):
        return v

    @field_validator("CURRTY")
    @classmethod
    def validate_currty(cls, v):
        return v

    @field_validator("WATLEV")
    @classmethod
    def validate_watlev(cls, v):
        return v

    @field_validator("WATLEVOLD")
    @classmethod
    def validate_watlevold(cls, v):
        return v

    @field_validator("DEP")
    @classmethod
    def validate_dep(cls, v):
        return v

    @field_validator("TAUW")
    @classmethod
    def validate_tauw(cls, v):
        return v

    @field_validator("TAUHF")
    @classmethod
    def validate_tauhf(cls, v):
        return v

    @field_validator("TAUTOT")
    @classmethod
    def validate_tautot(cls, v):
        return v

    @field_validator("STOKESSURFX")
    @classmethod
    def validate_stokessurfx(cls, v):
        return v

    @field_validator("STOKESSURFY")
    @classmethod
    def validate_stokessurfy(cls, v):
        return v

    @field_validator("STOKESBAROX")
    @classmethod
    def validate_stokesbarox(cls, v):
        return v

    @field_validator("STOKESBAROY")
    @classmethod
    def validate_stokesbaroy(cls, v):
        return v


class Station(NamelistBaseModel):
    BEGTC: Optional[str] = Field(
        "'19980901.000000'",
        description="Start simulation time in format 'yyyymmdd.hhmmss'. Must match the simulation time for output to be generated. If not specified, defaults to PROC%BEGTC.",
    )
    DELTC: Optional[list] = Field(
        [1, ""],
        description="Time step for output in seconds. If smaller than simulation time step, the simulation time step is used. Smaller values allow for better 1D and 2D spectra analysis.",
    )
    UNITC: Optional[str] = Field(
        "'SEC'",
        description="Unit of time for DELTC. Currently only supports 'SEC' for seconds.",
    )
    ENDTC: Optional[str] = Field(
        "'20081101.000000'",
        description="Stop time for simulation in format 'yyyymmdd.hhmmss'. If not specified, defaults to PROC%ENDC.",
    )
    OUTSTYLE: Optional[str] = Field(
        "'STE'",
        description="Output option. Use 'NO' to maximize efficiency during parallel runs using MPI. 'STE' is used for standard output.",
    )
    FILEOUT: Optional[str] = Field(
        "'station.dat'", description="Name of the output file for station data."
    )
    LOUTITER: Optional[str] = Field(
        "F",
        description="Boolean flag for iteration output (purpose not specified in given content).",
    )
    LLOUTS: Optional[str] = Field("F", description="Boolean flag for station output.")
    ILOUTS: Optional[int] = Field(1, description="Number of output stations.")
    NLOUTS: Optional[list] = Field(
        ["'P-1'", ""], description="Comma-separated list of names for output locations."
    )
    IOUTS: Optional[int] = Field(
        1,
        description="Index or identifier for output stations (purpose not clear from given content).",
    )
    NOUTS: Optional[list] = Field(
        ["'P-1'", ""],
        description="Name of output locations (purpose in relation to NLOUTS not clear from given content).",
    )
    XOUTS: Optional[list] = Field(
        [1950.0, ""], description="X-Coordinate of output locations."
    )
    YOUTS: Optional[list] = Field(
        [304.0, ""], description="Y-Coordinate of output locations."
    )
    CUTOFF: Optional[str] = Field(
        "8*0.44",
        description="Cutoff frequency (Hz) for each station, consistent with buoys. Specified as a list of 8 values.",
    )
    LSP1D: Optional[str] = Field(
        "F", description="Boolean flag for 1D spectral station output."
    )
    LSP2D: Optional[str] = Field(
        "F", description="Boolean flag for 2D spectral station output."
    )
    LSIGMAX: Optional[str] = Field(
        "T",
        description="Boolean flag to adjust the cut-off frequency for the output (e.g., to be consistent with buoy cut-off frequency).",
    )

    @field_validator("BEGTC")
    @classmethod
    def validate_begtc(cls, v):
        if not re.match(r"\d{8}\.\d{6}$", v):
            raise ValueError("BEGTC must be in format yyyymmdd.hhmmss")
        return v

    @field_validator("DELTC")
    @classmethod
    def validate_deltc(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("DELTC must be a positive integer")
        return v

    @field_validator("UNITC")
    @classmethod
    def validate_unitc(cls, v):
        if v != "SEC":
            raise ValueError("UNITC must be 'SEC'")
        return v

    @field_validator("ENDTC")
    @classmethod
    def validate_endtc(cls, v):
        if not re.match(r"\d{8}\.\d{6}$", v):
            raise ValueError("ENDTC must be in format yyyymmdd.hhmmss")
        return v

    @field_validator("OUTSTYLE")
    @classmethod
    def validate_outstyle(cls, v):
        if v not in ["STE", "NO"]:
            raise ValueError("OUTSTYLE must be either 'STE' or 'NO'")
        return v

    @field_validator("FILEOUT")
    @classmethod
    def validate_fileout(cls, v):
        if not v.endswith(".dat"):
            raise ValueError("FILEOUT must have a .dat extension")
        return v

    @field_validator("LOUTITER")
    @classmethod
    def validate_loutiter(cls, v):
        return bool(v)

    @field_validator("LLOUTS")
    @classmethod
    def validate_llouts(cls, v):
        return bool(v)

    @field_validator("ILOUTS")
    @classmethod
    def validate_ilouts(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("ILOUTS must be a positive integer")
        return v

    @field_validator("NLOUTS")
    @classmethod
    def validate_nlouts(cls, v):
        if not all(name.strip() for name in v.split(",")):
            raise ValueError("NLOUTS must not contain empty names")
        return v

    @field_validator("IOUTS")
    @classmethod
    def validate_iouts(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("IOUTS must be a positive integer")
        return v

    @field_validator("NOUTS")
    @classmethod
    def validate_nouts(cls, v):
        if not v.strip():
            raise ValueError("NOUTS must not be empty")
        return v

    @field_validator("XOUTS")
    @classmethod
    def validate_xouts(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("XOUTS must be a number")
        return v

    @field_validator("YOUTS")
    @classmethod
    def validate_youts(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("YOUTS must be a number")
        return v

    @field_validator("CUTOFF")
    @classmethod
    def validate_cutoff(cls, v):
        values = [float(x) for x in v.split("*")]
        if len(values) != 2 or values[0] != 8 or values[1] <= 0:
            raise ValueError('CUTOFF must be in format "8*<positive_float>"')
        return v

    @field_validator("LSP1D")
    @classmethod
    def validate_lsp1d(cls, v):
        return bool(v)

    @field_validator("LSP2D")
    @classmethod
    def validate_lsp2d(cls, v):
        return bool(v)

    @field_validator("LSIGMAX")
    @classmethod
    def validate_lsigmax(cls, v):
        return bool(v)

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
    KSPTYPE: Optional[str] = Field(
        "'bcgs'",
        description="Controls the linear solver algorithm used by PETSc. Options include GMRES, LGMRES, DGMRES, PGMRES, KSPBCGSL, and bcgs (BiCGStab). Each algorithm has specific characteristics suitable for different problem types.",
    )
    RTOL: Optional[str] = Field(
        "1.E-20",
        description="Relative convergence tolerance, representing the relative decrease in the residual norm required for convergence.",
    )
    ABSTOL: Optional[str] = Field(
        "1.E-20",
        description="Absolute convergence tolerance, representing the absolute size of the residual norm required for convergence.",
    )
    DTOL: Optional[float] = Field(
        10000.0,
        description="Divergence tolerance, used to detect divergence in the iterative solver.",
    )
    MAXITS: Optional[int] = Field(
        1000,
        description="Maximum number of iterations allowed for the iterative solver.",
    )
    INITIALGUESSNONZERO: Optional[str] = Field(
        "T",
        description="Boolean flag indicating whether the initial guess for the iterative solver is nonzero.",
    )
    GMRESPREALLOCATE: Optional[str] = Field(
        "T",
        description="Boolean flag to enable preallocation of all needed work vectors for GMRES and FGMRES at initial setup.",
    )
    PCTYPE: Optional[str] = Field(
        "'sor'",
        description="Controls the preconditioner type used by PETSc. Options include SOR, ASM, HYPRE, SPAI, and NONE. Each preconditioner is suitable for different problem types and solver configurations.",
    )

    @field_validator("KSPTYPE")
    @classmethod
    def validate_ksptype(cls, v):
        valid_types = ["bcgs", "GMRES", "LGMRES", "DGMRES", "PGMRES", "KSPBCGSL"]
        if v not in valid_types:
            raise ValueError(f"KSPTYPE must be one of {valid_types}")
        return v

    @field_validator("RTOL")
    @classmethod
    def validate_rtol(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("RTOL must be between 0 and 1 exclusively")
        return v

    @field_validator("ABSTOL")
    @classmethod
    def validate_abstol(cls, v):
        if v < 0:
            raise ValueError("ABSTOL must be non-negative")
        return v

    @field_validator("DTOL")
    @classmethod
    def validate_dtol(cls, v):
        if v <= 0:
            raise ValueError("DTOL must be positive")
        return v

    @field_validator("MAXITS")
    @classmethod
    def validate_maxits(cls, v):
        if v <= 0 or not isinstance(v, int):
            raise ValueError("MAXITS must be a positive integer")
        return v

    @field_validator("INITIALGUESSNONZERO")
    @classmethod
    def validate_initialguessnonzero(cls, v):
        return v

    @field_validator("GMRESPREALLOCATE")
    @classmethod
    def validate_gmrespreallocate(cls, v):
        return v

    @field_validator("PCTYPE")
    @classmethod
    def validate_pctype(cls, v):
        valid_types = ["sor", "SOR", "ASM", "HYPRE", "SPAI", "NONE"]
        if v not in valid_types:
            raise ValueError(f"PCTYPE must be one of {valid_types}")
        return v


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
