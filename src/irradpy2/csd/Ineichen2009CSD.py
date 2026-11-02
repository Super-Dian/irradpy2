import numpy as np
import matplotlib.pyplot as plt


def _ensure_column_vector(x, var_name):
    """
    Ensure input is a column vector.

    Parameters
    ----------
    x : array-like
        Input data.
    var_name : str
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        Array reshaped to (n, 1).

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


def Ineichen2009CSD(ghi, exth, zen, plot_figure=None):
    """
    Ineichen 2009 Clear Sky Detection model.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance, column vector.
    exth : array-like
        Horizontal projection of extraterrestrial irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    plot_figure : any, optional
        If not None, plot the detection result.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection flag as a column vector.
        1 means clouds were detected, 0 means clear sky.
    """
    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    ghi = _ensure_column_vector(ghi, "ghi")
    zen = _ensure_column_vector(zen, "zen")
    exth = _ensure_column_vector(exth, "exth")

    if len(np.unique([len(ghi), len(exth), len(zen)])) != 1:
        raise ValueError("vars must be equal in length")

    # ------------------------------------------------------------------
    # Clearness index (eq 7 in [1])
    # MATLAB: kt = ghi./exth;
    # ------------------------------------------------------------------
    kt = ghi / exth

    # ------------------------------------------------------------------
    # Optical air mass from [2]
    # MATLAB:
    # h = 90-zen;
    # M = 1./(sind(h) + 0.15.*(h + 3.885).^-1.253);
    # ------------------------------------------------------------------
    h = 90.0 - zen
    M = 1.0 / (np.sin(np.deg2rad(h)) + 0.15 * (h + 3.885) ** (-1.253))

    # ------------------------------------------------------------------
    # Modified global / surface irradiance clearness index
    # MATLAB:
    # kt_prime = kt ./  ( 1.031 .* exp(-1.4 ./ (0.9 + 9.4 ./ M))+0.1);
    # ------------------------------------------------------------------
    kt_prime = kt / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / M)) + 0.1)

    # ------------------------------------------------------------------
    # Perform the clear-sky detection criteria
    # MATLAB:
    # csd = zeros(size(ghi));
    # csd(kt_prime<0.65)=1;
    # ------------------------------------------------------------------
    csd = np.zeros_like(ghi)
    csd[kt_prime < 0.65] = 1

    # ------------------------------------------------------------------
    # Figure
    # MATLAB logic preserved:
    # if exist('plot_figure','var')
    # ------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="ineichen 2009 CSD example", facecolor="w")
        plt.plot(ghi.reshape(-1))
        CSD = ghi.copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD.reshape(-1), linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd