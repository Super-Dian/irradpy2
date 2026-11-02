import numpy as np
import matplotlib.pyplot as plt


def Zhang2018CSD(ghi: np.ndarray, ghics: np.ndarray, plot_figure=None) -> np.ndarray:
    """
    Zhang 2018 clear-sky detection model.

    References
    ----------
    Zhang, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018).
    Modeling and Simulation of High Frequency Solar Irradiance.
    Preprint to Journal of Photovoltaics.

    Zhang, W., Kleiber, W., Florita, A. R., Hodge, B.-M., and Mather, B. (2018).
    A stochastic downscaling approach for generating high-frequency solar irradiance scenarios.
    Solar Energy, 176, 370-379.

    Inputs
    ------
    ghi : np.ndarray
        Global horizontal irradiance [W m-2], column vector.
    ghics : np.ndarray
        Clear-sky global horizontal irradiance [W m-2], column vector.
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

    def _hankel_from_first_col_and_last_row(first_col: np.ndarray, last_row: np.ndarray) -> np.ndarray:
        """
        Build a Hankel matrix equivalent to MATLAB hankel(c, r).
        """
        first_col = np.asarray(first_col, dtype=float).reshape(-1)
        last_row = np.asarray(last_row, dtype=float).reshape(-1)
        values = np.concatenate([first_col, last_row[1:]])
        row_idx = np.arange(first_col.size)[:, None]
        col_idx = np.arange(last_row.size)[None, :]
        return values[row_idx + col_idx]

    # Safety checks
    ghi = _to_column_vector(ghi, "ghi")
    ghics = _to_column_vector(ghics, "ghics")

    if len(set([len(ghi), len(ghics)])) != 1:
        raise ValueError("vars must be equal in length")

    gamma_limit = 0.1

    # The methodology is applied to 30 min clear-sky GHI linearly interpolated back to 1 min resolution
    ghics30 = np.nanmean(_hankel_from_first_col_and_last_row(ghics.reshape(-1), np.concatenate(([ghics.reshape(-1)[-1]], np.full(29, np.nan)))), axis=1).reshape(-1, 1)
    ghi30 = np.nanmean(_hankel_from_first_col_and_last_row(ghi.reshape(-1), np.concatenate(([ghi.reshape(-1)[-1]], np.full(29, np.nan)))), axis=1).reshape(-1, 1)

    x = np.arange(1, len(ghics30) + 1, dtype=float)
    xx = np.linspace(1, len(ghics30), len(ghics))

    ghics30lin = np.interp(xx, x, ghics30.reshape(-1)).reshape(-1, 1)
    ghi30lin = np.interp(xx, x, ghi30.reshape(-1)).reshape(-1, 1)

    # Principal based on:
    # dghi(t) = ghi(t) - ghi(t-1)
    # dghics(t) = ghics(t) - ghics(t-1)
    # gamma = max(abs(dghi(t) - dghics(t)))
    # Where gamma is taken as a maximum for the day. Should it be greater than 0.1, the day is said to be cloudy
    dghi = np.vstack([np.nan, np.diff(ghi30lin.reshape(-1))[:, None]])
    dghics = np.vstack([np.nan, np.diff(ghics30lin.reshape(-1))[:, None]])
    gamma = np.abs(dghi - dghics)

    csd = np.zeros_like(ghi)
    csd[gamma > gamma_limit] = 1

    # Plot
    if plot_figure is not None:
        plt.figure(num="Zhang CSD example", facecolor="w")
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