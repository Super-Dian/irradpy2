import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin


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


def AliaMartinez2016CSD(ghi: np.ndarray,ghics: np.ndarray,zen: np.ndarray,datetime: np.ndarray,plot_figure=None):
    """
    Alia-Martinez 2016 clear-sky detection model.

    References
    ----------
    Alia-Martinez, M., Antonanzas, J., Urraca, R., Martinez-De-Pison, F. J.,
    and Antonanzas-Torres, F. (2016).
    Benchmark of algorithms for solar clear-sky detection.
    Journal of Renewable and Sustainable Energy, 8(3).

    Polo, J., Zarzalejo, L. F., Martín, L., Navarro, A. A., and Marchante, R. (2009).
    Estimation of daily Linke turbidity factor by using global irradiance
    measurements at solar noon. Solar Energy, 83(8), 1177-1185.

    Inputs
    ------
    ghi : np.ndarray
        Global horizontal irradiance [W m-2], column vector.
    ghics : np.ndarray
        Clear-sky global horizontal irradiance [W m-2], column vector.
    zen : np.ndarray
        Zenith angle [deg], column vector.
    datetime : np.ndarray
        Datetime input.
    plot_figure : optional
        If provided, plot the final clear-sky detection result.

    Outputs
    -------
    csd : np.ndarray
        Clear-sky detection flag.
        1 = cloudy
        0 = clear

    Notes
    -----
    This function is a strict Python translation of the MATLAB version.
    The following parts are intentionally preserved:
    - parameter names
    - parameter order
    - threshold values
    - decision order
    - intermediate logic
    """

    ghi = np.asarray(ghi)
    ghics = np.asarray(ghics)
    zen = np.asarray(zen)

    if ghi.ndim == 1:
        ghi = ghi.reshape(-1, 1)
    elif ghi.ndim == 2 and ghi.shape[0] == 1 and ghi.shape[1] != 1:
        ghi = ghi.T
    if ghi.ndim != 2 or ghi.shape[1] != 1:
        raise ValueError("var ghi must be a column vector")

    if ghics.ndim == 1:
        ghics = ghics.reshape(-1, 1)
    elif ghics.ndim == 2 and ghics.shape[0] == 1 and ghics.shape[1] != 1:
        ghics = ghics.T
    if ghics.ndim != 2 or ghics.shape[1] != 1:
        raise ValueError("var ghics must be a column vector")

    if zen.ndim == 1:
        zen = zen.reshape(-1, 1)
    elif zen.ndim == 2 and zen.shape[0] == 1 and zen.shape[1] != 1:
        zen = zen.T
    if zen.ndim != 2 or zen.shape[1] != 1:
        raise ValueError("var zen must be a column vector")

    Dayth, Year = datetime_convert(datetime)

    if len(np.unique([len(ghi), len(ghics), len(zen), len(Dayth)])) != 1:
        raise ValueError("vars must be equal in length")

    ghi = ghi[:, 0]
    ghics = ghics[:, 0]
    zen = zen[:, 0]

    threshold = 0.007
    slagdiff = 0.005
    maxtimepeak = 15
    maxtimetrans = 25

    dtn = Year * 1000 + Dayth
    unique_days = np.unique(dtn)

    csd_daily = np.ones(ghi.shape, dtype=float)
    csd_minute = np.ones(ghi.shape, dtype=float)

    alpha = 1.0
    previous_alpha = np.nan
    persisted_alpha = alpha

    for d in range(len(unique_days)):
        ind = np.where(dtn == unique_days[d])[0]

        GHI = ghi[ind]
        GHIcs = ghics[ind] * persisted_alpha

        # Test 1: only consider irradiance > 100 W m-2
        ind100 = np.where(GHI > 100)[0]

        GHIcs = GHIcs[ind100]
        GHI = GHI[ind100]

        if GHIcs.size != 0:
            # Test 2: relative change and slag
            relative = (GHIcs - GHI) / GHIcs
            slag = np.concatenate([np.diff(relative), np.array([np.nan])])

            ind_slag = np.where(np.abs(slag) < slagdiff)[0]
            delta_ind_slag = np.diff(ind_slag)

            if len(delta_ind_slag) > 1:
                # Test 3 and Test 4
                rel_at_pos = relative[ind_slag]
                relative_at_pos_mean = np.nanmean(
                    np.column_stack([rel_at_pos[:-1], rel_at_pos[1:]]),
                    axis=1
                )

                delete_mask = (delta_ind_slag < 10) & (relative_at_pos_mean > 0.3)

                if np.any(delete_mask):
                    keep_first_part = ~delete_mask
                    if len(ind_slag) > len(delta_ind_slag):
                        ind_slag = np.concatenate([
                            ind_slag[:len(delta_ind_slag)][keep_first_part],
                            ind_slag[len(delta_ind_slag):]
                        ])
                    else:
                        ind_slag = ind_slag[keep_first_part]

                    delta_ind_slag = delta_ind_slag[keep_first_part]

                if len(delta_ind_slag) > 1:
                    clear_sky = False

                    # Test 5 and Test 6: daily clear criteria
                    values, lengths = rle(delta_ind_slag)
                    peaks = lengths[values == 1]

                    if (peaks > maxtimepeak).size != 0:
                        clear_sky = True

                    if np.sum(peaks) > maxtimetrans:
                        clear_sky = True

                    if clear_sky:
                        GHIcs_for_optimising = GHIcs[ind_slag]
                        GHI_for_optimising = GHI[ind_slag]

                        previous_alpha = persisted_alpha
                        iter_count = 0
                        max_iterations = 10

                        while (round(alpha * 10000) != round(previous_alpha * 10000)) and (iter_count <= max_iterations):
                            previous_alpha = alpha

                            def rmse(a):
                                a = np.asarray(a).reshape(-1)[0]
                                return np.sqrt(np.nanmean((GHI_for_optimising - a * GHIcs_for_optimising) ** 2))

                            alpha = fmin(rmse, alpha, disp=False)[0]
                            iter_count = iter_count + 1

                        relative = (GHIcs * alpha - GHI) / (GHIcs * alpha)
                        slag = np.concatenate([np.diff(relative), np.array([np.nan])])
                        ind_slag = np.where(np.abs(slag) < slagdiff)[0]
                        delta_ind_slag = np.diff(ind_slag)
                        rel_at_pos = relative[ind_slag]

                        if len(rel_at_pos) >= 2:
                            relative_at_pos_mean = np.nanmean(
                                np.column_stack([rel_at_pos[:-1], rel_at_pos[1:]]),
                                axis=1
                            )
                            delete_mask = (delta_ind_slag < 10) & (relative_at_pos_mean > 0.3)

                            if np.any(delete_mask):
                                if len(ind_slag) > len(delta_ind_slag):
                                    ind_slag = np.concatenate([
                                        ind_slag[:len(delta_ind_slag)][~delete_mask],
                                        ind_slag[len(delta_ind_slag):]
                                    ])
                                else:
                                    ind_slag = ind_slag[~delete_mask]

                        if len(ind_slag) > 0:
                            csd_minute[ind[ind100[ind_slag]]] = 0

                        GHIcs = GHIcs[ind_slag] * alpha
                        GHI = GHI[ind_slag]

                    if len(GHIcs) >= 2 and len(GHI) >= 2:
                        corr_matrix = np.corrcoef(GHIcs, GHI)
                        if np.linalg.det(corr_matrix) <= threshold:
                            csd_daily[ind] = 0
                            persisted_alpha = alpha

    csd = np.ones(ghi.shape, dtype=float)
    csd[(csd_minute == 0) & (csd_daily == 0)] = 0

    if plot_figure is not None:
        plt.figure(num="Alia-Martinez CSD example", facecolor="w")
        plt.plot(ghi)
        CSD = ghi.astype(float).copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [W m-2]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return np.asarray(csd, dtype=float).reshape(-1, 1)


def rle(Input):
    """
    Run length encoding helper.

    Inputs
    ------
    Input : array-like
        One-dimensional input array.

    Outputs
    -------
    values : ndarray
        Run values.
    lengths : ndarray
        Run lengths.

    Notes
    -----
    This helper is also written to preserve the MATLAB implementation logic.
    """

    Input = np.asarray(Input).flatten()
    L = len(Input)

    if L == 0:
        return np.array([]), np.array([])

    j = 0
    k = 0
    i = 0

    Output = np.zeros(2 * L,dtype=Input.dtype if np.issubdtype(Input.dtype, np.number) else object)

    while i < 2 * L:
        comp = 1
        for j in range(j, L):
            if j == L - 1:
                break
            if Input[j] == Input[j + 1]:
                comp = comp + 1
            else:
                break

        Output[k + 1] = comp
        Output[k] = Input[j]

        if j == L - 1 and L > 1 and Input[j - 1] == Input[j]:
            break

        i = i + 1
        k = k + 2
        j = j + 1

        if j == L - 1:
            if k + 1 < len(Output):
                Output[k + 1] = 1
                Output[k] = Input[j]
            break

        if j >= L:
            break

    lengths = Output[1::2]
    values = Output[0:-1:2]

    lengths = lengths[lengths != 0]
    values = values[:len(lengths)]

    return values, lengths