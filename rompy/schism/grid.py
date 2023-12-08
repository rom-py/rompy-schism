import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import Field, field_validator, model_validator
from pyschism.mesh import Hgrid
from shapely.geometry import MultiPoint, Polygon

from rompy.core import DataBlob
from rompy.core.grid import BaseGrid

logger = logging.getLogger(__name__)


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
    manning: Optional[DataBlob] = Field(
        default=None, description="Path to manning.gr3 file"
    )
    hgridll: Optional[DataBlob] = Field(
        default=None, description="Path to hgrid.ll file"
    )
    diffmin: Optional[DataBlob] = Field(
        default=None, description="Path to diffmax.gr3 file"
    )
    diffmax: Optional[DataBlob] = Field(
        default=None, description="Path to diffmax.gr3 file"
    )
    hgrid_WWM: Optional[DataBlob] = Field(
        default=None, description="Path to hgrid_WWM.gr3 file"
    )
    wwmbnd: Optional[DataBlob] = Field(
        default=None, description="Path to wwmbnd.gr3 file"
    )

    @model_validator(mode="after")
    def validate_rough_drag_manning(cls, v):
        if sum([v.rough is not None, v.drag is not None, v.manning is not None]) > 1:
            raise ValueError("Only one of rough, drag, manning can be set")
        return v

    def _set_xy(self):
        """Set x and y coordinates from hgrid."""

        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        self.x = hgrid.x
        self.y = hgrid.y

    def get(self, staging_dir: Path):
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
        ]:
            source = getattr(self, filetype)
            if source is not None:
                logger.info(
                    f"Copying {source.id}: {source.source} to {staging_dir}/{source.source}"
                )
                source.get(staging_dir)
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
        from cartopy import crs as ccrs
        from matplotlib.tri import Triangulation

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(121)
        ax.set_title("Bathymetry")

        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        hgrid.make_plot(axes=ax)

        # open boundary nodes/info as geopandas df
        gdf_open_boundary = hgrid.boundaries.open

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

        meshtri = Triangulation(hgrid.x, hgrid.y, hgrid.elements.array)
        ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
        ax.triplot(meshtri, color="k", alpha=0.3)
        gdf_open_boundary.plot(ax=ax, color="b")
        ax.add_geometries(
            hgrid.boundaries.land.geometry.values,
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
        ax.set_title("Mesh")

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

    def get(self, staging_dir: Path):
        ret = super().get(staging_dir)
        for filetype in [
            "vgrid",
        ]:
            source = getattr(self, filetype)
            if source is not None:
                logger.info(
                    f"Copying {source.id}: {source.source} to {staging_dir}/{source.source}"
                )
                source.get(staging_dir)
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
    grid.plot_hgrid()
    # plt.figure()
    # grid._set_xy()
    # bnd = grid.boundary()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # # plot polygon on cartopy axes
    # ax.add_geometries([bnd], ccrs.PlateCarree(), facecolor="none", edgecolor="red")
    # ax.coastlines()
    plt.show()
