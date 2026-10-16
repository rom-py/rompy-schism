import os

from pydantic import Field

from rompy_schism.namelists.basemodel import NamelistBaseModel


class Sflux_Inputs(NamelistBaseModel):
    """``sflux/sflux_inputs.txt``: the sflux files and variable names SCHISM reads.

    Written by SCHISMDataSflux from its sources; not set in a configuration.
    """

    air_1_relative_weight: float = Field(
        1.0,
        description="air_[12]_relative_weight set the relative ratio between datasets '1' and '2'",
    )
    air_2_relative_weight: float = Field(
        99.0,
        description="air_[12]_relative_weight set the relative ratio between datasets '1' and '2'",
    )
    air_1_max_window_hours: float = Field(
        120.0,
        description="max. # of hours (offset from start time in each file) in each file of set '1'",
    )
    air_2_max_window_hours: float = Field(
        120.0,
        description="max. # of hours (offset from start time in each file) in each file of set '1'",
    )
    air_1_fail_if_missing: bool = Field(True, description="set '1' is mandatory")
    air_2_fail_if_missing: bool = Field(False, description="set '2' is optional")
    air_1_file: str = Field("air_1", description="file name for 1st set of 'air'")
    air_2_file: str = Field("air_2", description="file name for 2nd set of 'air'")
    uwind_name: str = Field("uwind", description="name of u-wind vel.")
    vwind_name: str = Field("vwind", description="name of v-wind vel.")
    prmsl_name: str = Field(
        "prmsl", description="name of air pressure (@MSL) variable in .nc file"
    )
    stmp_name: str = Field("stmp", description="name of surface air T")
    spfh_name: str = Field("spfh", description="name of specific humidity")
    rad_1_relative_weight: float = Field(
        1.0, description="relative weight of set '1' of rad when two sets are given"
    )
    rad_2_relative_weight: float = Field(
        99.0, description="relative weight of set '2' of rad when two sets are given"
    )
    rad_1_max_window_hours: float = Field(
        24.0,
        description="max. # of hours (offset from start time in each file) in each file of set '1'",
    )
    rad_2_max_window_hours: float = Field(
        24.0,
        description="max. # of hours (offset from start time in each file) in each file of set '2'",
    )
    rad_1_fail_if_missing: bool = Field(
        False, description="stop if set '1' of rad is missing"
    )
    rad_2_fail_if_missing: bool = Field(
        False, description="stop if set '2' of rad is missing"
    )
    rad_1_file: str = Field("rad_1", description="file name of set '1' of rad")
    rad_2_file: str = Field("rad_2", description="file name of set '2' of rad")
    dlwrf_name: str = Field(
        "dlwrf", description="name of downward long wave radiation variable"
    )
    dswrf_name: str = Field(
        "dswrf", description="name of downward short wave radiation variable"
    )
    prc_1_relative_weight: float = Field(
        1.0, description="relative weight of set '1' of prc when two sets are given"
    )
    prc_2_relative_weight: float = Field(
        99.0, description="relative weight of set '2' of prc when two sets are given"
    )
    prc_1_max_window_hours: float = Field(
        24.0,
        description="max. # of hours (offset from start time in each file) in each file of set '1'",
    )
    prc_2_max_window_hours: float = Field(
        24.0,
        description="max. # of hours (offset from start time in each file) in each file of set '2'",
    )
    prc_1_fail_if_missing: bool = Field(
        False, description="stop if set '1' of prc is missing"
    )
    prc_2_fail_if_missing: bool = Field(
        False, description="stop if set '2' of prc is missing"
    )
    prc_1_file: str = Field("prc_1", description="file name of set '1' of prc")
    prc_2_file: str = Field("prc_2", description="file name of set '2' of prc")
    prate_name: str = Field("prate", description="name of precipitation rate variable")
    priority: int = Field(1, description="priority of the data")

    def render(self) -> str:
        """Render the namelist variable as a string"""
        # create string of the form "variable = value"
        ret = []
        ret += ["! SCHISM sflux namelist rendered from Rompy"]
        ret += ["&sflux_inputs"]
        for key, value in self.model_dump().items():
            if key in ["id", "priority", "file_number"]:
                continue
            if isinstance(value, list):
                value = ", ".join([str(item) for item in value])
            if isinstance(value, str):
                value = f"'{value}'"
            if isinstance(value, bool):
                value = ".true." if value else ".false."
            ret += [f"{key}={value},"]
        ret += ["/\n"]
        return "\n".join(ret)

    def write_nml(self, destdir: str) -> None:
        """Write the namelist to a file"""
        os.makedirs(destdir, exist_ok=True)
        output = destdir / f"{self.__class__.__name__.lower()}.txt"
        with open(output, "w") as f:
            f.write(self.render())
        return output
