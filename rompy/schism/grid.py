import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import Field, field_validator, model_validator
from pyschism.mesh import Hgrid
from shapely.geometry import MultiPoint, Polygon

from rompy.core import DataBlob, RompyBaseModel
from rompy.core.grid import BaseGrid

logger = logging.getLogger(__name__)


import os

from pydantic import BaseModel, field_validator


class GR3Generator(RompyBaseModel):
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    gr3_type: str = Field(
        ...,
        description="Type of gr3 file. Must be one of 'albedo', 'diffmin', 'diffmax', 'watertype', 'windrot_geo2proj'",
    )
    val: float = Field(None, description="Constant value to set in gr3 file")

    @field_validator("gr3_type")
    def gr3_type_validator(cls, v):
        if v not in [
            "albedo",
            "diffmin",
            "diffmax",
            "watertype",
            "windrot_geo2proj",
        ]:
            raise ValueError(
                "gr3_type must be one of 'albedo', 'diffmin', 'diffmax', 'watertype', 'windrot_geo2proj'"
            )
        return v

    @property
    def id(self):
        return self.gr3_type

    def generate_gr3(self, destdir: str | Path) -> Path:
        if isinstance(self.hgrid, DataBlob):
            if not self.hgrid._copied:
                self.hgrid.get(destdir)
            ref = self.hgrid._copied
        else:
            ref = self.hgrid
        dest = Path(destdir) / f"{self.gr3_type}.gr3"
        with open(ref, "r") as inFile:
            with open(dest, "w") as outFile:
                # blank line
                line = inFile.readline()
                outFile.write(f"{self.val}\n")

                # ne, np
                line = inFile.readline()
                outFile.write(line)

                ne, np1 = map(int, line.strip().split())

                for i in range(np1):
                    line = inFile.readline().strip()
                    id, x, y, z = map(float, line.split())
                    outFile.write(f"{id} {x} {y} {self.val}\n")

                for i in range(ne):
                    line = inFile.readline()
                    outFile.write(line)
                return dest


# TODO - check datatypes for gr3 files (int vs float)
class SCHISMGrid2D(BaseGrid):
    """2D SCHISM grid in geographic space."""

    grid_type: Literal["2D", "3D"] = Field(
        "2D", description="Type of grid (2D=two dimensional, 3D=three dimensional)"
    )
    hgrid: DataBlob = Field(..., description="Path to hgrid.gr3 file")
    drag: Optional[DataBlob] = Field(default=None, description="Path to drag.gr3 file")
    rough: Optional[DataBlob] = Field(
        default=None, description="Path to rough.gr3 file"
    )
    manning: Optional[DataBlob | int] = Field(
        default=None, description="Path to manning.gr3 file"
    )
    hgridll: Optional[DataBlob | int] = Field(
        default=None, description="Path to hgrid.ll file"
    )
    diffmin: Optional[DataBlob | float] = Field(
        default=1.0e-6,
        description="Path to diffmax.gr3 file or constant value",
        validate_default=True,
    )
    diffmax: Optional[DataBlob | float] = Field(
        default=1.0,
        description="Path to diffmax.gr3 file or constant value",
        validate_default=True,
    )
    albedo: Optional[DataBlob | float] = Field(
        default=0.15,
        description="Path to albedo.gr3 file or constant value",
        validate_default=True,
    )
    watertype: Optional[DataBlob | int] = Field(
        default=1,
        description="Path to watertype.gr3 file or constant value",
        validate_default=True,
    )
    windrot_geo2proj: Optional[DataBlob | float] = Field(
        default=0.0,
        description="Path to windrot_geo2proj.gr3 file or constant value",
        validate_default=True,
    )
    hgrid_WWM: Optional[DataBlob | int] = Field(
        default=None, description="Path to hgrid_WWM.gr3 file"
    )
    wwmbnd: Optional[DataBlob | int] = Field(
        default=None, description="Path to wwmbnd.gr3 file"
    )

    @model_validator(mode="after")
    def validate_rough_drag_manning(cls, v):
        if sum([v.rough is not None, v.drag is not None, v.manning is not None]) > 1:
            raise ValueError("Only one of rough, drag, manning can be set")
        return v

    # validator that checks for gr3 source and if if not a daba blob or GR3Input, initializes a GR3Generator
    @field_validator(
        "drag",
        "rough",
        "manning",
        "hgridll",
        "diffmin",
        "diffmax",
        "albedo",
        "watertype",
        "windrot_geo2proj",
        "hgrid_WWM",
        "wwmbnd",
        mode="after",
    )
    def gr3_source_validator(cls, v, values):
        if v is not None:
            if not isinstance(v, DataBlob):
                v = GR3Generator(
                    hgrid=values.data["hgrid"], gr3_type=values.field_name, val=v
                )
        return v

    def _set_xy(self):
        """Set x and y coordinates from hgrid."""

        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        self.x = hgrid.x
        self.y = hgrid.y

    def get(self, destdir: Path) -> dict:
        ret = {}
        for filetype in [
            "hgrid",
            "drag",
            "rough",
            "manning",
            "hgridll",
            "diffmin",
            "diffmax",
            "hgrid_WWM",
            "wwmbnd",
            "albedo",
            "watertype",
            "windrot_geo2proj",
        ]:
            source = getattr(self, filetype)
            if source is not None:
                if isinstance(source, DataBlob):
                    logger.info(
                        f"Copying {source.id}: {source.source} to {destdir}/{source.source}"
                    )
                    source.get(destdir)
                else:
                    logger.info(
                        f"Generating {source.id}: {source.gr3_type} with value {source.val}"
                    )
                    source.generate_gr3(destdir)
        return ret

    def _get_boundary(self, tolerance=None) -> Polygon:
        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        bnd = pd.concat(
            [
                hgrid.boundaries.open.get_coordinates(),
                hgrid.boundaries.land.get_coordinates(),
            ]
        )
        # convert pandas dataframe to polygon
        polygon = Polygon(zip(bnd.x.values, bnd.y.values))
        if tolerance:
            polygon = polygon.simplify(tolerance=tolerance)
        return polygon

    def plot_hgrid(self):
        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        hgrid.make_plot()

    def ocean_boundary(self):
        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        bnd = hgrid.boundaries.open.get_coordinates()
        return bnd.x.values, bnd.y.values

    def land_boundary(self):
        from pyschism.mesh import Hgrid

        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        bnd = hgrid.boundaries.land.get_coordinates()
        return bnd.x.values, bnd.y.values

    # def boundary_points(self, tolerance=0.2):
    #     """
    #     Convenience function to convert boundary Shapely Polygon
    #     to arrays of coordinates
    #
    #     Parameters
    #     ----------
    #     tolerance : float
    #         Passed to Grid.boundary
    #         See https://shapely.readthedocs.io/en/stable/manual.html#object.simplify
    #
    #     """
    #     return self.ocean_boundary()


class SCHISMGrid3D(SCHISMGrid2D):
    """3D SCHISM grid in geographic space."""

    grid_type: Literal["2D", "3D"] = Field(
        "3D", description="Type of grid (2D=two dimensional, 3D=three dimensional)"
    )
    vgrid: DataBlob = Field(..., description="Path to vgrid.in file")

    def get(self, destdir: Path):
        ret = super().get(destdir)
        for filetype in [
            "vgrid",
        ]:
            source = getattr(self, filetype)
            if source is not None:
                logger.info(
                    f"Copying {source.id}: {source.source} to {destdir}/{source.source}"
                )
                source.get(destdir)
        return ret


if __name__ == "__main__":
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    grid = SCHISMGrid2D(
        hgrid=DataBlob(
            source="../../tests/schism/test_data/hgrid.gr3",
            id="hgrid",
        )
    )
    # grid.plot_hgrid()
    # plt.figure()
    # grid._set_xy()
    # bnd = grid.boundary()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # # plot polygon on cartopy axes
    # ax.add_geometries([bnd], ccrs.PlateCarree(), facecolor="none", edgecolor="red")
    # ax.coastlines()
    # plt.show()
