"""Filling the missing values of ocean model data.

Ocean models have no values below their seabed or on land. Before their data are
interpolated to SCHISM's nodes and levels, which can lie deeper than the ocean
model's seabed or next to its land, those gaps are filled so the interpolation does
not mix in values from other depths or places.
"""

import numpy as np
from scipy import ndimage


def fill_below_seabed(values: np.ndarray, axis: int) -> np.ndarray:
    """Extend each profile down from its deepest valid value.

    Parameters
    ----------
    values : np.ndarray
        The data, with depth increasing along `axis`.
    axis : int
        The depth axis.

    Returns
    -------
    np.ndarray
        A copy of the data with the gaps below each profile filled. Profiles with
        no valid values are left missing.
    """
    values = np.moveaxis(np.array(values, dtype=float), axis, 0)
    for k in range(1, values.shape[0]):
        missing = np.isnan(values[k])
        values[k][missing] = values[k - 1][missing]
    return np.moveaxis(values, 0, axis)


def fill_land(values: np.ndarray) -> np.ndarray:
    """Fill the columns with no valid values from the nearest column that has some.

    Parameters
    ----------
    values : np.ndarray
        The data, shaped (..., y, x); a column is missing if its first value is.

    Returns
    -------
    np.ndarray
        A copy of the data with land columns filled.
    """
    values = np.array(values, dtype=float)
    land = np.isnan(values.reshape(-1, *values.shape[-2:])[0])
    if land.all():
        raise ValueError("The ocean data have no valid values")
    if land.any():
        iy, ix = ndimage.distance_transform_edt(
            land, return_distances=False, return_indices=True
        )
        values = values[..., iy, ix]
    return values
