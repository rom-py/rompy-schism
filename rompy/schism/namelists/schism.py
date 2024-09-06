import logging
from pathlib import Path
from typing import Optional

from pydantic import Field

from rompy.core import TimeRange
from rompy.schism.namelists.basemodel import NamelistBaseModel

from .cosine import Cosine
from .example import Example
from .ice import Ice
from .icm import Icm
from .mice import Mice
from .param import Param
from .sediment import Sediment
from .wwminput_spectra import Wwminput as Wwminput_specstra
from .wwminput_WW3 import Wwminput as Wwminput_ww3

logger = logging.getLogger(__name__)


class NML(NamelistBaseModel):
    param: Optional[Param] = Field(description="Model paramaters", default=None)
    ice: Optional[Ice] = Field(description="Ice model parameters", default=None)
    icm: Optional[Icm] = Field(description="Ice model parameters", default=None)
    mice: Optional[Mice] = Field(description="Ice model parameters", default=None)
    sediment: Optional[Sediment] = Field(
        description="Sediment model parameters", default=None
    )
    cosine: Optional[Cosine] = Field(
        description="Sediment model parameters", default=None
    )
    wmminput_spectra: Optional[Wwminput_specstra] = Field(
        description="Wave model input parameters", default=None
    )
    wwminput_WW3: Optional[Wwminput_ww3] = Field(
        description="Wave model input parameters", default=None
    )

    def update_times(self, period=TimeRange):
        """
        This class is used to set consistent time parameters in a group component by
        redefining existing `times` component attribute based on the `period` field.

        """

        update = {
            "param": {
                "opt": {
                    "start_year": period.start.year,
                    "start_month": period.start.month,
                    "start_day": period.start.day,
                    "start_hour": period.start.hour,
                    "rnday": period.duration.total_seconds() / 86400,
                }
            }
        }

        date_format = "%Y-%m-%d %H:%M:%S"
        if hasattr(
            self, "wwminput_WW3"
        ):  # TODO change this check to the actual flag value
            # TODO these are currently all the same, but they could be different
            update.update(
                {
                    "wwminput_WW3": {
                        "proc": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "wind": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "curr": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "walv": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "station": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "hotfile": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                    }
                }
            )
        self.update(update)

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
                attr.write_nml(workdir)


if __name__ == "__main__":
    nml = NML()
    nml.write_nml(Path("test"))
