from rompy.core.types import RompyBaseModel


class NamelistBaseModel(RompyBaseModel):
    """Base model for namelist variables"""

    def render(self) -> str:
        """Render the namelist variable as a string"""
        # create string of the form "variable = value"
        ret = []
        ret += [f"! SCHISM {self.__module__} namelist rendered from Rompy\n"]
        for section, values in self.model_dump().items():
            ret += [f"&{section}"]
            for variable, value in values.items():
                for ii in range(13):
                    variable = variable.replace(f"__{ii}", f"({ii})")
                if isinstance(value, list):
                    value = ", ".join([str(item) for item in value])
                if isinstance(value, bool):
                    value = ".true." if value else ".false."
                if isinstance(value, str):
                    value = f"'{value}'"
                ret += [f"{variable} = {value}"]
            ret += ["/"]
        return "\n".join(ret)

    def write_nml(self, workdir: str) -> None:
        """Write the namelist to a file"""
        output = workdir / f"{self.__class__.__name__.lower()}.nml"
        with open(output, "w") as f:
            f.write(self.render())
