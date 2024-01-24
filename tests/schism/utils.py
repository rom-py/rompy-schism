import logging
from datetime import datetime
from glob import glob
from pathlib import Path
from time import time

import xarray as xr
from matplotlib.transforms import Bbox
from pyschism.forcing.hycom.hycom2schism import DownloadHycom
from pyschism.mesh.hgrid import Hgrid

"""
Download hycom data for Fortran scripts.
Default is to download data for generating initial condition (use hgrid 
    as parameter to cover the whole domain).
Optionally, download data for generating th.nc (bnd=True) and nu.nc (nudge=True) 
    (use bbox as parameter to cut small domain).
"""
logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger("pyschism")
logger.setLevel(logging.INFO)


def download_hycom(dest=Path("./"), hgrid=Path("./hgrid.gr3")):
    hgrid = Hgrid.open(hgrid, crs="epsg:4326")
    date = datetime(2023, 1, 1)
    hycom = DownloadHycom(hgrid=hgrid)
    t0 = time()
    hycom.fetch_data(date, rnday=2, bnd=False, nudge=False, sub_sample=5, outdir="./")
    print(f"It took {(time()-t0)/60} mins to download")

    files = glob("hycom_*.nc")
    files.sort()
    logging.info("Concatenating files...")
    xr.open_mfdataset(files, concat_dim="time", combine="nested")["surf_el"].to_netcdf(
        dest / "hycom.nc"
    )
    for file in files:
        os.remove(file)


def compare_files(file1, file2):
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            for line1, line2 in zip(f1, f2):
                if line1[0] != "$" and line2[0] != "$":
                    assert line1.rstrip() == line2.rstrip()
