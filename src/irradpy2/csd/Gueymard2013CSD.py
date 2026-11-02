import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel


def _ensure_column_vector(x, var_name):
    """
    Ensure the input is a column vector.

    Parameters
    ----------
    x : array-like
        Input data.
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
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2 or arr.shape[1] != 1:
        arr = arr.T
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"var {var_name} must be a column vector")

    return arr


def Gueymard2013CSD(dni, dnics, zen, plot_figure=None):
    """
    Gueymard 2013 clear-sky detection method.

    References
    ----------
    C.A. Gueymard. 2013.
    Aerosol turbidity derivation from broadband irradiance measurements:
    methodological advances and uncertainty analysis.
    ASES Solar 2013 Conf., Baltimore, MD (2013)

    Parameters
    ----------
    dni : array-like
        Direct normal irradiance, column vector.
    dnics : array-like
        Clear-sky direct normal irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    plot_figure : any, optional
        If provided, a diagnostic figure is plotted.

    Returns
    -------
    csd : np.ndarray
        Clear-sky detection flag as a column vector.
        1 = cloudy
        0 = clear
    """
    # ------------------------------------------------------------------ #
    # Safety checks
    # ------------------------------------------------------------------ #
    dni = _ensure_column_vector(dni, "dni")
    dnics = _ensure_column_vector(dnics, "dnics")
    zen = _ensure_column_vector(zen, "zen")

    if len(np.unique([len(dni), len(dnics)])) != 1:
        raise ValueError("vars must be equal in length")

    # ------------------------------------------------------------------ #
    # Parametrisation
    # ------------------------------------------------------------------ #
    Tranmn1 = 0.98
    Tranmn2 = 0.9
    Varia = 0.015
    Zmax = 85
    DNImin = 120

    # ------------------------------------------------------------------ #
    # Pre-allocation
    # 1 = cloudy. Base assumption unless proven otherwise
    # ------------------------------------------------------------------ #
    csd = np.ones_like(dni, dtype=float)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    Tran_current = dni / dnics

    """
    MATLAB logic:
    Tran_hankel = hankel(Tran_current,[Tran_current(end),NaN]);

    This builds a 2-column structure effectively giving:
    column 1: Tran(t)
    column 2: Tran(t+1)

    Then:
    var2 = abs(col2/col1 - 1)  -> described in original code as t/(t-1)
    var1 = abs(col1/col2 - 1)  -> described in original code as t/(t+1)

    Here we preserve the exact implemented numerical behavior.
    """
    tran_flat = Tran_current.reshape(-1)
    Tran_hankel = hankel(tran_flat, np.array([tran_flat[-1], np.nan], dtype=float))

    var2 = np.abs(Tran_hankel[:, 1] / Tran_hankel[:, 0] - 1).reshape(-1, 1)
    var1 = np.abs(Tran_hankel[:, 0] / Tran_hankel[:, 1] - 1).reshape(-1, 1)

    # ------------------------------------------------------------------ #
    # Apply all 3 criteria
    # A period is clear if all three criteria are met.
    # ------------------------------------------------------------------ #

    # criteria 1: DNI(t) > DNImin
    c1 = dni > DNImin

    # criteria 2: zenith(t) < Zmax
    c2 = zen < Zmax

    # criteria 3: stability check with the original three logical branches
    c3 = (
        (Tran_current > Tranmn1)
        | ((Tran_current > Tranmn2) & ((var2 < Varia) & (var1 < Varia)))
        | ((var2 < Varia) & (Tran_current > 0.95))
    )

    # assert that csd of all agreeing criteria are clear
    csd[c1 & c2 & c3] = 0

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    if plot_figure is not None:
        plt.figure(num="Gueymard CSD example", facecolor="w")
        plt.plot(dni.reshape(-1))
        CSD = dni.reshape(-1).copy()
        CSD[csd.reshape(-1) == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k")
        plt.legend(["GHI", "Clear detected"])
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time")
        plt.show()

    return csd