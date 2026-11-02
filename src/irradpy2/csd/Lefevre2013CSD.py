import numpy as np
import matplotlib.pyplot as plt


def _ensure_column_vector(x, var_name):
    """
    Ensure the input is a column vector, matching the original MATLAB
    safety-check logic.

    Parameters
    ----------
    x : array-like
        Input vector.
    var_name : str
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        Column vector with shape (n, 1).

    Raises
    ------
    ValueError
        If the input cannot be converted into a column vector.
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
    Build a Hankel-style trailing window matrix consistent with the MATLAB code:
    hankel(x, [x(end), NaN(1, window_size-1)])

    Each row starts at the current index and extends forward.

    Parameters
    ----------
    x : np.ndarray
        Column vector or 1D vector.
    window_size : int
        Window length.

    Returns
    -------
    np.ndarray
        Array of shape (n, window_size).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)

    out = np.full((n, window_size), np.nan, dtype=float)
    for j in range(window_size):
        end_idx = n - j
        if end_idx > 0:
            out[:end_idx, j] = x[j:j + end_idx]

    return out


def Lefevre2013CSD(ghi, dhi, exth, zen, plot_figure=None):
    """
    Lefevre 2013 Clear Sky Detection model.

    Structural Python refactor of the original MATLAB implementation.
    Parameters, thresholds, logic order, and output meaning are preserved.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance, column vector.
    dhi : array-like
        Diffuse horizontal irradiance, column vector.
    exth : array-like
        Horizontal projection of extraterrestrial irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    plot_figure : any, optional
        If defined, a figure illustrating the result is plotted.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection flag as a column vector.
        1 means clouds were detected.
        0 means clear sky.
    """
    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    ghi = _ensure_column_vector(ghi, "ghi")
    dhi = _ensure_column_vector(dhi, "dhi")
    zen = _ensure_column_vector(zen, "zen")
    exth = _ensure_column_vector(exth, "exth")

    if len(np.unique([len(ghi), len(dhi), len(zen), len(exth)])) != 1:
        raise ValueError("vars must be equal in length")

    ghi_1d = ghi.reshape(-1)
    dif_1d = dhi.reshape(-1)
    zen_1d = zen.reshape(-1)
    exth_1d = exth.reshape(-1)

    # ------------------------------------------------------------------
    # Pre-allocation
    # From [1]; "Two filters have been applied on the remaining BSRN data
    # order to retain reliable clear-sky instants."
    # ------------------------------------------------------------------
    csd = np.ones_like(ghi_1d, dtype=float)

    # ------------------------------------------------------------------
    # First filter: diffuse to global ratio
    # From [1]; retain only minutes with Ediff/Eglo < 0.3
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        k = dif_1d / ghi_1d

    csd[k < 0.3] = 0

    # ------------------------------------------------------------------
    # Second filter: temporal variability based on corrected clearness index
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi_1d / exth_1d

    h = 90.0 - zen_1d
    with np.errstate(divide="ignore", invalid="ignore"):
        M = 1.0 / (np.sin(np.deg2rad(h)) + 0.15 * np.power(h + 3.885, -1.253))

    # Same formula as the MATLAB implementation
    with np.errstate(divide="ignore", invalid="ignore"):
        kt_prime = kt / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / M)) + 0.1)

    # 90-minute forward Hankel window, matching:
    # std(hankel(kt_prime,[kt_prime(end),NaN(1,89)]),0,2,'omitnan')
    KTp_window = _hankel_window(kt_prime, 90)
    KTp = np.nanstd(KTp_window, axis=1, ddof=1)

    # Apply the test such that clear is only identified if KTp < 0.02
    csd[KTp < 0.02] = 0

    csd = csd.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="Lefevre CSD example", facecolor="w")
        plt.plot(ghi_1d)
        CSD = ghi_1d.copy()
        CSD[csd.reshape(-1) == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd