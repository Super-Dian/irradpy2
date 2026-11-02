import numpy as np
import matplotlib.pyplot as plt


def Xie2013CSD(ghi: np.ndarray, ghics: np.ndarray, dni: np.ndarray, dnics: np.ndarray, zen: np.ndarray, exth: np.ndarray, plot_figure=None) -> np.ndarray:
    """
    Xie 2013 clear-sky detection model.

    References
    ----------
    Xie, Y., and Liu, Y. (2013).
    A new approach for simultaneously retrieving cloud albedo and cloud fraction from surface-based shortwave radiation measurements.
    Environmental Research Letters, 8(044023), 1-9.

    Inputs
    ------
    ghi : np.ndarray
        Global horizontal irradiance [W m-2], column vector.
    ghics : np.ndarray
        Clear-sky global horizontal irradiance [W m-2], column vector.
    dni : np.ndarray
        Direct normal irradiance [W m-2], column vector.
    dnics : np.ndarray
        Clear-sky direct normal irradiance [W m-2], column vector.
    zen : np.ndarray
        Zenith angle [deg], column vector.
    exth : np.ndarray
        Horizontal projection of extraterrestrial irradiance, column vector.
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
    - output definition
    """

    def _to_column_vector(arr: np.ndarray, var_name: str) -> np.ndarray:
        """
        Convert input to a column vector following MATLAB-style checks.
        """
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            return arr.reshape(-1, 1)

        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr
            if arr.shape[0] == 1:
                return arr.T
            raise ValueError(f"var {var_name} must be a column vector")

        raise ValueError(f"var {var_name} must be a column vector")

    # Safety checks
    ghi = _to_column_vector(ghi, "ghi")
    ghics = _to_column_vector(ghics, "ghics")
    dni = _to_column_vector(dni, "dni")
    dnics = _to_column_vector(dnics, "dnics")
    zen = _to_column_vector(zen, "zen")
    exth = _to_column_vector(exth, "exth")

    if len(set([len(ghi), len(ghics), len(dni), len(dnics), len(zen), len(exth)])) != 1:
        raise ValueError("vars must be equal in length")

    # F_dn_all is the all-sky downwelling shortwave radiative flux at surface
    F_all_dn = ghi
    # F_dn_all is the clear-sky downwelling shortwave radiative flux at surface
    F_clr_dn = ghics
    # F_all_d_dn is the all-sky downwelling of shortwave direct radiative flux at surface
    F_all_d_dn = dni
    # F_clr_d_dn is the clear-sky downwelling of shortwave direct radiative flux at surface
    F_clr_d_dn = dnics
    # Solar zenith
    mu = zen
    # F0 is extraterrestrial horizontal irradiance
    F0 = exth

    # Surface albedo
    salb = 0.2

    F_all_up = salb * F_all_dn

    A1 = mu * mu * F0 * F0 * (F_clr_dn - F_all_dn)
    A2 = F_clr_dn * mu * mu * F0 * F0 - F_all_up * F_clr_dn * F_clr_dn
    B1 = A1 / A2
    B2 = (F_clr_d_dn - F_all_d_dn) / F_clr_d_dn

    # Assign re
    re = np.zeros_like(B2)
    x = B1 / B2
    product_B1_B2 = B1 * B2

    mask0 = (product_B1_B2 == 0) | ((product_B1_B2 > 0.07) & (product_B1_B2 < 0.07872))
    re[mask0] = 0

    mask_nan = x < 0.07
    re[mask_nan] = np.nan

    mask1 = (x >= 0.07872) & (x < 0.11442)
    re[mask1] = 1.0 - 31.16480 * x[mask1] + np.sqrt((31.1648 * x[mask1]) ** 2 - 49.6255 * x[mask1])

    mask2 = (x > 0.11442) & (x < 0.185)
    re[mask2] = (2.61224 * x[mask2] - 1.0 + np.sqrt(24.2004 * x[mask2] ** 2 - 9.0098 * x[mask2] + 1.0)) / (18.3622 * x[mask2] - 4.0)

    mask3 = (x > 0.185) & (x < 0.23792)
    re[mask3] = 0.89412 * x[mask3] + 0.02519

    mask4 = x > 0.23792
    re[mask4] = x[mask4]

    test1 = np.abs(F_clr_dn - F_all_dn) / F_clr_dn
    test2 = (F_clr_dn - F_all_dn) / F_clr_dn
    test3 = np.abs(F_clr_d_dn - F_all_d_dn) / F_clr_d_dn
    test4 = (F_clr_d_dn - F_all_d_dn) / F_clr_d_dn
    re[((test1 < 0.05) & (test2 < 0.0)) | ((test3 < 0.05) & (test4 < 0.0))] = 0

    fraction1 = B1 / re
    fraction1[re == 0] = 0

    csd = np.zeros_like(fraction1)
    csd[fraction1 != 0] = 1

    # Plot
    if plot_figure is not None:
        plt.figure(num="Xie CSD example", facecolor="w")
        plt.plot(ghi, label="GHI")
        clear_detected = ghi.astype(float).copy()
        clear_detected[csd == 1] = np.nan
        plt.plot(clear_detected, linewidth=2, color="k", label="Clear detected")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.legend()
        plt.show()

    return csd