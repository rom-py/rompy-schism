import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from pyschism.mesh import Hgrid
from pyschism.mesh.prop import Tvdflag
from pyschism.mesh.vgrid import Vgrid
from shapely.geometry import MultiPoint, Polygon

from rompy.core import DataBlob, RompyBaseModel
from rompy.core.grid import BaseGrid

logger = logging.getLogger(__name__)


import os

from pydantic import BaseModel, field_validator
from pyschism.mesh.hgrid import Gr3


class GR3Generator(RompyBaseModel):
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    gr3_type: str = Field(
        ...,
        description="Type of gr3 file. Must be one of 'albedo', 'diffmin', 'diffmax', 'watertype', 'windrot_geo2proj'",
    )
    value: float = Field(None, description="Constant value to set in gr3 file")
    crs: str = Field("epsg:4326", description="Coordinate reference system")

    @field_validator("gr3_type")
    def gr3_type_validator(cls, v):
        accept = ["albedo", "diffmin", "diffmax", "watertype", "windrot_geo2proj"]
        warn = ["manning", "rough", "drag"]
        if v not in accept + warn:
            raise ValueError(
                "gr3_type must be one of 'albedo', 'diffmin', 'diffmax', 'watertype', 'windrot_geo2proj'"
            )
        if v in warn:
            logger.warning(
                f"{v} is being set to a constant value, this is not recommended. For best results, please supply friction gr3 files with spatially varying values. Further options are under development."
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
        hgrid = Gr3.open(ref, crs=self.crs)
        grd = hgrid.copy()
        grd.description = self.gr3_type
        grd.nodes.values[:] = self.value
        grd.write(dest, overwrite=True)
        logger.info(f"Generated {self.gr3_type} with constant value of {self.value}")
        return dest

    def get(self, destdir: str | Path) -> Path:
        """Alias to maintain api compatibility with DataBlob"""
        return self.generate_gr3(destdir)


# TODO - check datatypes for gr3 files (int vs float)
class SCHISMGrid(BaseGrid):
    """2D SCHISM grid in geographic space."""

    grid_type: Literal["schism"] = Field("schism", description="Model descriminator")
    hgrid: DataBlob = Field(
        ..., description="Path to hgrid.gr3 file"
    )  # TODO always need. Follow up with Vanessa
    vgrid: Optional[DataBlob] = Field(default=None, description="Path to vgrid.in file")
    drag: Optional[DataBlob | float] = Field(
        default=None, description="Path to drag.gr3 file"
    )
    rough: Optional[DataBlob | float] = Field(
        default=None, description="Path to rough.gr3 file"
    )
    manning: Optional[DataBlob | float] = Field(
        default=None,
        description="Path to manning.gr3 file",  # TODO treat in the same way as the other gr3 files. Add a warning that this is not advisable
    )
    hgridll: Optional[DataBlob | int] = Field(
        default=None, description="Path to hgrid.ll file. "
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
        default=None,
        description="Path to wwmbnd.gr3 file",  # This is generated on the fly. Script sent from Vanessa.
    )
    crs: str = Field("epsg:4326", description="Coordinate reference system")
    _pyschism_hgrid: Optional[Hgrid] = None
    _pyschism_vgrid: Optional[Vgrid] = None

    @model_validator(mode="after")
    def validate_rough_drag_manning(cls, v):
        fric_sum = sum([v.rough is not None, v.drag is not None, v.manning is not None])
        if fric_sum > 1:
            raise ValueError("Only one of rough, drag, manning can be set")
        if fric_sum == 0:
            raise ValueError("At least one of rough, drag, manning must be set")
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
                    hgrid=values.data["hgrid"], gr3_type=values.field_name, value=v
                )
        return v

    @model_validator(mode="after")
    def set_xy(cls, v):
        if v.hgrid is not None:
            v._pyschism_hgrid = Hgrid.open(v.hgrid._copied or v.hgrid.source, crs=v.crs)
            v.x = v._pyschism_hgrid.x
            v.y = v._pyschism_hgrid.y
        return v

    @property
    def pyschism_hgrid(self):
        if self._pyschism_hgrid is None:
            self._pyschism_hgrid = Hgrid.open(
                self.hgrid._copied or self.hgrid.source, crs=self.crs
            )
        return self._pyschism_hgrid

    @property
    def pyschism_vgrid(self):
        if self.vgrid is None:
            return None
        if self._pyschism_vgrid is None:
            self._pyschism_vgrid = Vgrid.open(self.vgrid._copied or self.vgrid.source)
        return self._pyschism_vgrid

    @property
    def is_3d(self):
        if self.vgrid is not None:
            return True
        else:
            return False

    def get(self, destdir: Path) -> dict:
        ret = {}
        for filetype in [
            "hgrid",
            "vgrid",
            "drag",
            "rough",
            "manning",
            "diffmin",
            "diffmax",
            "hgridll",
            "hgrid_WWM",
            "wwmbnd",
            "albedo",
            "watertype",
            "windrot_geo2proj",
        ]:
            source = getattr(self, filetype)
            if filetype == "hgridll":
                if source is None:
                    logger.info(f"Creating symbolic link for hgrid.ll")
                    os.symlink("./hgrid.gr3", f"{destdir}/hgrid.ll")
                    continue
            if source is not None:
                source.get(destdir)
        self.generate_tvprop(destdir)
        return ret

    def generate_tvprop(self, destdir: Path) -> Path:
        """Generate tvprop.in file

        Args:
            destdir (Path): Destination directory

        Returns:
            Path: Path to tvprop.in file
        """
        # TODO - should this be handled in the same way as the gr3 files? i.e. would you
        # ever want to provide a file path to tvprop.in?
        tvdflag = Tvdflag(
            self.pyschism_hgrid, np.array([1] * len(self.pyschism_hgrid.elements))
        )
        tvdflag.write(destdir / "tvprop.in")
        return destdir / "tvprop.in"

    def boundary(self, tolerance=None) -> Polygon:
        bnd = self.pyschism_hgrid.boundaries.open.get_coordinates()
        polygon = Polygon(zip(bnd.x.values, bnd.y.values))
        if tolerance:
            polygon = polygon.simplify(tolerance=tolerance)
        return polygon

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from cartopy import crs as ccrs
        from matplotlib.tri import Triangulation

        if ax is None:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        else:
            fig = plt.gcf()

        meshtri = Triangulation(
            self.pyschism_hgrid.x,
            self.pyschism_hgrid.y,
            self.pyschism_hgrid.elements.array,
        )
        ax.triplot(meshtri, color="k", alpha=0.3)

        # open boundary nodes/info as geopandas df
        gdf_open_boundary = self.pyschism_hgrid.boundaries.open

        # make a pandas dataframe for easier lon/lat referencing during forcing condition generation
        df_open_boundary = pd.DataFrame(
            {
                "schism_index": gdf_open_boundary.index_id[0],
                "index": gdf_open_boundary.indexes[0],
                "lon": gdf_open_boundary.get_coordinates().x,
                "lat": gdf_open_boundary.get_coordinates().y,
            }
        ).reset_index(drop=True)

        # create sub-sampled wave boundary
        # #wave_boundary = redistribute_vertices(gdf_open_boundary.geometry[0], 0.2)
        #
        # df_wave_boundary = pd.DataFrame(
        #     {"lon": wave_boundary.xy[0], "lat": wave_boundary.xy[1]}
        # ).reset_index(drop=True)
        gdf_open_boundary.plot(ax=ax, color="b")
        ax.add_geometries(
            self.pyschism_hgrid.boundaries.land.geometry.values,
            facecolor="none",
            edgecolor="g",
            linewidth=2,
            crs=ccrs.PlateCarree(),
        )
        ax.plot(
            df_open_boundary["lon"],
            df_open_boundary["lat"],
            "+k",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.plot(
            df_open_boundary["lon"],
            df_open_boundary["lat"],
            "xr",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        # ax.plot(
        #     df_wave_boundary["lon"],
        #     df_wave_boundary["lat"],
        #     "+k",
        #     transform=ccrs.PlateCarree(),
        #     zorder=10,
        # )
        # ax.plot(
        #     df_wave_boundary["lon"],
        #     df_wave_boundary["lat"],
        #     "xr",
        #     transform=ccrs.PlateCarree(),
        #     zorder=10,
        # )
        ax.coastlines()
        return fig, ax

    def plot_hgrid(self):
        import matplotlib.pyplot as plt
        from cartopy import crs as ccrs
        from matplotlib.tri import Triangulation

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(121)
        ax.set_title("Bathymetry")

        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        self.pyschism_hgrid.make_plot(axes=ax)

        ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
        self.plot(ax=ax)
        ax.set_title("Mesh")

    def ocean_boundary(self):
        bnd = self.pyschism_hgrid.boundaries.open.get_coordinates()
        return bnd.x.values, bnd.y.values

    def land_boundary(self):
        bnd = self.pyschism_hgrid.boundaries.land.get_coordinates()
        return bnd.x.values, bnd.y.values


if __name__ == "__main__":
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    grid = SCHISMGrid2D(
        hgrid=DataBlob(
            source="../../tests/schism/test_data/hgrid.gr3",
            id="hgrid",
        )
    )
    grid.plot_hgrid()
    # plt.figure()
    # grid._set_xy()
    # bnd = grid.boundary()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # # plot polygon on cartopy axes
    # ax.add_geometries([bnd], ccrs.PlateCarree(), facecolor="none", edgecolor="red")
    # ax.coastlines()
    plt.show()
