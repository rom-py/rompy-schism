from pathlib import Path
from typing import Optional

from pydantic import Field

from rompy.schism.namelists.basemodel import NamelistBaseModel

from .cosine import COSINE
from .example import EXAMPLE
from .ice import ICE
from .icm import ICM
from .mice import MICE
from .param import PARAM
from .sediment import SEDIMENT
from .wwminput_spectra import WWMINPUT as WWMINPUT_SPECSTRA
from .wwminput_WW3 import WWMINPUT as WWMINPUT_WW3


class NML(NamelistBaseModel):
    param: PARAM = Field(description="Model paramaters")
    ice: Optional[ICE] = Field(description="Ice model parameters", default=None)
    icm: Optional[ICM] = Field(description="Ice model parameters", default=None)
    mice: Optional[MICE] = Field(description="Ice model parameters", default=None)
    sediment: Optional[SEDIMENT] = Field(
        description="Sediment model parameters", default=None
    )
    cosine: Optional[COSINE] = Field(
        description="Sediment model parameters", default=None
    )
    wmminput_spectra: Optional[WWMINPUT_SPECSTRA] = Field(
        description="Wave model input parameters", default=None
    )
    wwminput_WW3: Optional[WWMINPUT_WW3] = Field(
        description="Wave model input parameters", default=None
    )

    def write_nml(self, workdir: Path):
        for nml in [
            "param",
            "ice",
            "icm",
            "mice",
            "sediment",
            "cosine",
            "wmminput_spectra",
            "wwminput_WW3",
        ]:
            attr = getattr(self, nml)
            if attr is not None:
                try:
                    attr.write_nml(workdir)
                except Exception as e:
                    __import__("ipdb").set_trace()


if __name__ == "__main__":
    nml = NML()
    nml.write_nml(Path("test"))
