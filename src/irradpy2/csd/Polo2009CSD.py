import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def datetime_convert(datetime: np.ndarray):
    """Convert a datetime sequence to (day-of-year, year) arrays.

    Parameters
    ----------
    datetime : np.ndarray
        Datetime sequence parsed with ``pandas.to_datetime``; invalid
        entries become NaT. The original array shape is preserved.

    Returns
    -------
    Dayth : numpy.ndarray of float
        Day of year (1-366), same shape as ``datetime``.
    Year : numpy.ndarray of float
        Calendar year, same shape as ``datetime``.
    """
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def Polo2009CSD(ghi: np.ndarray, ghics: np.ndarray, zen: np.ndarray, datetime: np.ndarray, plot_figure=None):
    """
    Polo 2009 clear-sky detection model.

    References
    ----------
    J. Polo, L. F. Zarzalejo, L. Martín, A. A. Navarro, and R. Marchante,
    ‘Good CSDs methodology, not a good CSDc, measurements at solar noon’,
    Sol. Energy, vol. 83, no. 8, pp. 1177–1185, 2009.

    Inputs
    ------
    ghi : np.ndarray
        Global horizontal irradiance, column vector. Not necessary to be continuous so long as the ghics perfectly corresponds.
    ghics : np.ndarray
        Clear-sky global horizontal irradiance, column vector.
    zen : np.ndarray
        Zenith angle in degrees. Corresponding to all inputs.
    datetime : np.ndarray
        Datetime input.
    plot_figure : optional
        If provided, a figure is plotted illustrating the outcome of the CSD method, e.g. 1.

    Outputs
    -------
    csd : np.ndarray
        Clear-sky detection. A flag of 1 means that clouds were detected whereas a flag of 0 means that the hour is clear.
    """

    # safety checks
    ghi = np.array(ghi)
    ghics = np.array(ghics)
    zen = np.array(zen)

    if not _is_column_vector(ghi):
        ghi = ghi.T
        if not _is_column_vector(ghi):
            raise ValueError("var ghi must be a column vector")

    if not _is_column_vector(ghics):
        ghics = ghics.T
        if not _is_column_vector(ghics):
            raise ValueError("var ghics must be a column vector")

    if not _is_column_vector(zen):
        zen = zen.T
        if not _is_column_vector(zen):
            raise ValueError("var zen must be a column vector")

    Dayth, Year = datetime_convert(datetime)
    Dayth = Dayth.reshape(-1)
    Year = Year.reshape(-1)

    if len(np.unique(np.array([len(ghi), len(ghics), len(zen), len(Dayth)]))) != 1:
        raise ValueError("vars must be equal in length")

    # flatten to MATLAB-like column-vector behavior
    ghi = np.array(ghi).reshape(-1)
    ghics = np.array(ghics).reshape(-1)
    zen = np.array(zen).reshape(-1)

    # find the unique days
    dtn = Year * 1000 + Dayth
    unique_days = np.unique(dtn)

    csd = np.ones(ghi.shape)

    for i in range(len(unique_days)):
        # find data points of current day
        ind = np.where(dtn == unique_days[i])[0]

        # Determinant of correlation matrix
        determinant = np.linalg.det(np.corrcoef(ghi[ind], ghics[ind]))

        # only consider day clear if missing <80% of data points and det<0.005
        if len(ind) > 0.2 * 24 * 60 and determinant < 0.005:
            # is clear
            csd[ind] = 0

    # figure example
    if plot_figure is not None:
        plt.figure(num="Polo CSD example", facecolor="w")
        plt.plot(ghi)
        CSD = np.array(ghi, dtype=float)
        CSD[csd == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return np.asarray(csd, dtype=float).reshape(-1, 1)


def _is_column_vector(x: np.ndarray):
    """
    Check whether x behaves like a MATLAB column vector.
    """
    x = np.array(x)
    if x.ndim == 1:
        return True
    if x.ndim == 2 and x.shape[1] == 1:
        return True
    return False