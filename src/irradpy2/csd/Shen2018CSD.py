import numpy as np
import matplotlib.pyplot as plt


def Shen2018CSD(dni: np.ndarray,dnics: np.ndarray,aod550: np.ndarray,plot_figure=None) -> np.ndarray:
    """
    Shen 2018 clear-sky detection model.

    References
    ----------
    Shen, Y., Wei, H., Zhu, T., Zhao, X., and Zhang, K. (2018).
    A Data-driven Clear Sky Model for Direct Normal Irradiance.
    IOP Conference Series: Journal of Physics: Conference Series, 1072.

    Inputs
    ------
    dni : np.ndarray
        Direct normal irradiance [W m-2], column vector.
    dnics : np.ndarray
        Clear-sky direct normal irradiance [W m-2], column vector.
    aod550 : np.ndarray
        Total aerosol extinction at 550 nm, column vector.
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
    - parameter meaning
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

    # ---------------------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------------------
    dni = _to_column_vector(dni, "dni")
    dnics = _to_column_vector(dnics, "dnics")
    aod550 = _to_column_vector(aod550, "aod550")

    if len(set([len(dni), len(dnics), len(aod550)])) != 1:
        raise ValueError("vars must be equal in length")

    # ---------------------------------------------------------------------
    # Estimate aerosol extinction coefficient threshold
    # ---------------------------------------------------------------------
    test1 = np.zeros_like(dni)
    aod_limit = 0.5
    test1[aod550 < aod_limit] = 1

    # ---------------------------------------------------------------------
    # Estimate convexity
    # ---------------------------------------------------------------------
    noise_limit = 10  # Wm-2
    window_size = 10
    tolerance = 1

    dni_flat = dni.reshape(-1)
    dnics_flat = dnics.reshape(-1)

    dniw = _hankel_from_first_col_and_last_row(dni_flat, np.concatenate(([dni_flat[-1]], np.full(window_size - 1, np.nan))))
    dnicsw = _hankel_from_first_col_and_last_row(dnics_flat, np.concatenate(([dnics_flat[-1]], np.full(window_size - 1, np.nan))))

    rate_dni = ((dniw[:, -1] - dniw[:, 0]) / window_size).reshape(-1, 1)
    rate_dnics = ((dnicsw[:, -1] - dnicsw[:, 0]) / window_size).reshape(-1, 1)

    x = np.arange(window_size, dtype=float).reshape(1, -1)
    ydni = rate_dni * x + dniw[:, [0]]

    test_temp1 = np.repeat(np.abs(rate_dni - rate_dnics) < tolerance, window_size, axis=1).astype(int)
    test_temp2 = ((dniw + noise_limit) > ydni).astype(int)
    test_temp3 = (dniw > 50).astype(int)

    test_temp = test_temp1 + test_temp2 + test_temp3
    test2 = np.sum(test_temp, axis=1).reshape(-1, 1)
    test2[test2 < 3 * window_size] = 0
    test2[test2 == 3 * window_size] = 1

    # ---------------------------------------------------------------------
    # Build inverse clear-sky flag
    # ---------------------------------------------------------------------
    csd_inverse = np.zeros_like(dni)
    csd_inverse[(test1 + test2) == 2] = 1

    # ---------------------------------------------------------------------
    # Estimate duration
    # ---------------------------------------------------------------------
    duration_threshold = 30  # hour
    if duration_threshold % 2 != 0:
        raise ValueError("duration_threshold must be even")

    csd_inverse_flat = csd_inverse.reshape(-1)
    durations = np.sum(_hankel_from_first_col_and_last_row(csd_inverse_flat, np.concatenate(([csd_inverse_flat[-1]], np.full(duration_threshold - 1, np.nan)))), axis=1).reshape(-1, 1)

    durations = np.vstack([np.full((duration_threshold // 2, 1), np.nan), durations[:len(durations) - duration_threshold // 2]])

    # ---------------------------------------------------------------------
    # Pre-allocate output
    # ---------------------------------------------------------------------
    csd = np.ones_like(dni)

    # If duration_threshold consecutive points are clear, mark as clear
    csd[durations == duration_threshold] = 0

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(dni, label="DNI")

        clear_detected = dni.astype(float).copy()
        clear_detected[csd == 1] = np.nan
        plt.plot(clear_detected, linewidth=2, label="Clear detected")

        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xlabel("Time")
        plt.xticks([])
        plt.legend()
        plt.title("Shen CSD example")
        plt.tight_layout()
        plt.show()

    return csd