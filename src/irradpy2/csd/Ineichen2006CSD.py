import numpy as np
import matplotlib.pyplot as plt


def _ensure_column_vector(x, var_name):
    """
    Ensure input is a column vector, following the original MATLAB safety-check logic.

    Parameters
    ----------
    x : array-like
        Input data.
    var_name : str
        Variable name used in the error message.

    Returns
    -------
    np.ndarray
        Column vector with shape (n, 1).

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a column vector.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.ndim != 2 or x.shape[1] != 1:
        x = x.T
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(f"var {var_name} must be a column vector")

    return x


def Ineichen2006CSD(exth, dni, zen, plot_figure=None):
    """
    Ineichen 2006 clear-sky detection model.

    This is a structural Python refactor of the original MATLAB implementation.
    Parameters, thresholds, criteria order, and plotting logic are preserved.

    Parameters
    ----------
    exth : array-like
        Horizontal projection of extraterrestrial irradiance, column vector.
    dni : array-like
        Direct normal irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    plot_figure : any, optional
        If provided (not None), plot the detection result.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection flag as a column vector.
        0 means clear sky, 1 means clouds detected.
    """
    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    dni = _ensure_column_vector(dni, "dni")
    zen = _ensure_column_vector(zen, "zen")
    exth = _ensure_column_vector(exth, "exth")

    if len(np.unique([len(dni), len(zen), len(exth)])) != 1:
        raise ValueError("vars must be equal in length")

    # ------------------------------------------------------------------
    # Pre-allocate the CSD time series for use in criteria
    # MATLAB: csd = zeros(size(dni));
    # ------------------------------------------------------------------
    csd = np.zeros_like(dni)

    # ------------------------------------------------------------------
    # Air mass from Kasten 1980
    # MATLAB:
    # h = 90-zen;
    # M = 1./(sind(h) + 0.15.*(h + 3.885).^-1.253);
    # ------------------------------------------------------------------
    h = 90.0 - zen
    M = 1.0 / (np.sin(np.deg2rad(h)) + 0.15 * (h + 3.885) ** (-1.253))

    # ------------------------------------------------------------------
    # Ineichen's global horizontal terrestrial-irradiance
    # MATLAB:
    # alpha_k = M.*(9.4+0.9.*M).^-1;
    # gsc = exth.*exp(-2.*alpha_k);
    # ------------------------------------------------------------------
    alpha_k = M * (9.4 + 0.9 * M) ** (-1)
    gsc = exth * np.exp(-2.0 * alpha_k)

    # ------------------------------------------------------------------
    # Criterion 1
    # MATLAB:
    # csd(dni<0.9.*gsc)=1;
    # ------------------------------------------------------------------
    csd[dni < 0.9 * gsc] = 1

    # ------------------------------------------------------------------
    # Criterion 2
    # MATLAB:
    # delta_dni = (dni(1:end-2)+dni(3:end))./dni(2:end-1);
    # delta_dni = [NaN;delta_dni;NaN];
    #
    # delta_gsc = (gsc(1:end-2)+gsc(3:end))./gsc(2:end-1);
    # delta_gsc = [NaN;delta_gsc;NaN];
    #
    # csd(abs(delta_gsc-delta_dni)>=0.1)=1;
    # ------------------------------------------------------------------
    delta_dni_mid = (dni[:-2] + dni[2:]) / dni[1:-1]
    delta_dni = np.vstack((np.array([[np.nan]]), delta_dni_mid, np.array([[np.nan]])))

    delta_gsc_mid = (gsc[:-2] + gsc[2:]) / gsc[1:-1]
    delta_gsc = np.vstack((np.array([[np.nan]]), delta_gsc_mid, np.array([[np.nan]])))

    csd[np.abs(delta_gsc - delta_dni) >= 0.1] = 1

    # ------------------------------------------------------------------
    # Figure
    # MATLAB logic preserved:
    # if exist('plot_figure','var')
    # ------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="Ineichen 2006 CSD example", facecolor="w")
        plt.plot(dni.reshape(-1))
        CSD = dni.copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD.reshape(-1), linewidth=2, color="k")
        plt.legend(["DNI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd