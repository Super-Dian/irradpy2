import numpy as np
import matplotlib.pyplot as plt


def _ensure_column_vector(x, var_name):
    """
    Ensure input is a column vector.

    Parameters
    ----------
    x : array-like
        Input array.
    var_name : str
        Variable name used in error message.

    Returns
    -------
    np.ndarray
        Column vector with shape (n, 1).

    Raises
    ------
    ValueError
        If the input cannot be converted to a column vector.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.ndim != 2 or x.shape[1] != 1:
        x = x.T
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(f"var {var_name} must be a column vector")

    return x


def _hankel_window(x, window_size):
    """
    Reproduce MATLAB hankel(x,[x(end),NaN(1,window_size-1)]) behavior.

    Parameters
    ----------
    x : np.ndarray
        Column vector of shape (n, 1).
    window_size : int
        Window size.

    Returns
    -------
    np.ndarray
        Window matrix of shape (n, window_size).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)

    out = np.full((n, window_size), np.nan, dtype=float)
    for j in range(window_size):
        end_idx = n - j
        if end_idx > 0:
            out[:end_idx, j] = x[j:]

    return out


def Inman2015CSD(ghi, ghics, dni, dnics, plot_figure=None):
    """
    Inman 2015 Clear Sky Detection model.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance, column vector.
    ghics : array-like
        Clear-sky global horizontal irradiance, column vector.
    dni : array-like
        Direct normal irradiance, column vector.
    dnics : array-like
        Clear-sky direct normal irradiance, column vector.
    plot_figure : any, optional
        If provided, a figure illustrating the outcome is plotted.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection flag as a column vector.
        1 means that clouds were detected.
        0 means clear sky.
    """
    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    ghi = _ensure_column_vector(ghi, "ghi")
    ghics = _ensure_column_vector(ghics, "ghics")
    dni = _ensure_column_vector(dni, "dni")
    dnics = _ensure_column_vector(dnics, "dnics")

    if len(np.unique([len(ghi), len(ghics), len(dni), len(dnics)])) != 1:
        raise ValueError("vars must be equal in length")

    # ------------------------------------------------------------------
    # Inman thresholds
    # ------------------------------------------------------------------
    thresholds_ghi = np.array([100, 100, 50, 0.01, 10.0], dtype=float)
    thresholds_dni = np.array([200, 200, 100, 0.015, 15], dtype=float)

    window_size = 10

    # ------------------------------------------------------------------
    # Moving Window Vectorisation
    # ------------------------------------------------------------------
    ghi_window = _hankel_window(ghi, window_size)
    ghics_window = _hankel_window(ghics, window_size)
    dni_window = _hankel_window(dni, window_size)
    dnics_window = _hankel_window(dnics, window_size)

    # ------------------------------------------------------------------
    # Criteria analysis
    # clearsky criteria (five items per subcomponent)
    # ------------------------------------------------------------------
    cg1 = np.ones_like(ghi, dtype=float)
    cg2 = np.ones_like(ghi, dtype=float)
    cg3 = np.ones_like(ghi, dtype=float)
    cg4 = np.ones_like(ghi, dtype=float)
    cg5 = np.ones_like(ghi, dtype=float)

    cd1 = np.ones_like(ghi, dtype=float)
    cd2 = np.ones_like(ghi, dtype=float)
    cd3 = np.ones_like(ghi, dtype=float)
    cd4 = np.ones_like(ghi, dtype=float)
    cd5 = np.ones_like(ghi, dtype=float)

    # 1) average irradiance
    cg1[np.abs(np.nanmean(ghi_window, axis=1, keepdims=True) -
               np.nanmean(ghics_window, axis=1, keepdims=True)) < thresholds_ghi[0]] = 0

    cd1[np.abs(np.nanmean(dni_window, axis=1, keepdims=True) -
               np.nanmean(dnics_window, axis=1, keepdims=True)) < thresholds_dni[0]] = 0

    # 2) max irradiance
    cg2[np.abs(np.nanmax(ghi_window, axis=1, keepdims=True) -
               np.nanmax(ghics_window, axis=1, keepdims=True)) < thresholds_ghi[1]] = 0

    cd2[np.abs(np.nanmax(dni_window, axis=1, keepdims=True) -
               np.nanmax(dnics_window, axis=1, keepdims=True)) < thresholds_dni[1]] = 0

    # 3) irradiance increment
    diff_ghi = ghi_window[:, 1:] - ghi_window[:, :-1]
    diff_dni = dni_window[:, 1:] - dni_window[:, :-1]
    diff_ghics = ghics_window[:, 1:] - ghics_window[:, :-1]
    diff_dnics = dnics_window[:, 1:] - dnics_window[:, :-1]

    cg3[np.abs(np.nansum(np.sqrt(diff_ghi ** 2), axis=1, keepdims=True) -
               np.nansum(np.sqrt(diff_ghics ** 2), axis=1, keepdims=True)) < thresholds_ghi[2]] = 0

    cd3[np.abs(np.nansum(np.sqrt(diff_dni ** 2), axis=1, keepdims=True) -
               np.nansum(np.sqrt(diff_dnics ** 2), axis=1, keepdims=True)) < thresholds_dni[2]] = 0

    # 4) std of irradiance increment
    cg4[np.abs(
        np.nanstd(diff_ghi, axis=1, ddof=1, keepdims=True) / np.nanmean(ghi_window, axis=1, keepdims=True) -
        np.nanstd(diff_ghics, axis=1, ddof=1, keepdims=True) / np.nanmean(ghics_window, axis=1, keepdims=True)
    ) < thresholds_ghi[3]] = 0

    cd4[np.abs(
        np.nanstd(diff_dni, axis=1, ddof=1, keepdims=True) / np.nanmean(dni_window, axis=1, keepdims=True) -
        np.nanstd(diff_dnics, axis=1, ddof=1, keepdims=True) / np.nanmean(dnics_window, axis=1, keepdims=True)
    ) < thresholds_dni[3]] = 0

    # 5) max irradiance increment
    cg5[np.nanmax(np.abs(diff_ghi - diff_ghics), axis=1, keepdims=True) < thresholds_ghi[4]] = 0
    cd5[np.nanmax(np.abs(diff_dni - diff_dnics), axis=1, keepdims=True) < thresholds_dni[4]] = 0

    # sum of the criteria (0=all criteria met)
    criteria_ghi = cg1 + cg2 + cg3 + cg4 + cg5
    criteria_dni = cd1 + cd2 + cd3 + cd4 + cd5

    # both GHI and DNI tests must corroborate in order for csd to be detected
    criteria = criteria_ghi + criteria_dni

    # outputs
    csd = np.zeros_like(ghi, dtype=float)
    csd[criteria != 0] = 1

    # ------------------------------------------------------------------
    # Figure example
    # ------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="Inman CSD example", facecolor="w")
        plt.plot(ghi.reshape(-1))
        CSD = ghi.copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD.reshape(-1), linewidth=2, color="k")
        plt.legend(["GHI", "CSD"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time [min]")
        plt.show()

    return csd