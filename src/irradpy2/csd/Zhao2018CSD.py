import numpy as np
import matplotlib.pyplot as plt


def Zhao2018CSD(dni: np.ndarray, dnics: np.ndarray, plot_figure=None) -> np.ndarray:
    """
    Zhao 2018 clear-sky detection model.

    References
    ----------
    Zhao, X., Wei, H., Shen, Y., and Zhang, K. (2018).
    Real-time Clear-sky Model and Cloud Cover for Direct Normal Irradiance Prediction.
    IOP Conference Series: Journal of Physics: Conference Series, 1072.

    Inputs
    ------
    dni : np.ndarray
        Direct normal irradiance [W m-2], column vector.
    dnics : np.ndarray
        Clear-sky direct normal irradiance [W m-2], column vector.
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
    The following are intentionally preserved:
    - parameter names
    - parameter order
    - threshold values
    - algorithm flow
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

    def _nanmean_reshape_10min(arr: np.ndarray) -> np.ndarray:
        """
        Reproduce MATLAB logic:
        nanmean(reshape(arr(1:end-mod(end,10)), [10, N/10]), 1)

        MATLAB reshape is column-major, so order='F' must be used here.
        """
        n = len(arr)
        usable_len = n - (n % 10)
        arr_cut = arr[:usable_len].reshape(-1)
        arr_2d = np.reshape(arr_cut, (10, usable_len // 10), order="F")
        return np.nanmean(arr_2d, axis=0)

    # Safety checks
    dni = _to_column_vector(dni, "dni")
    dnics = _to_column_vector(dnics, "dnics")

    if len(set([len(dni), len(dnics)])) != 1:
        raise ValueError("vars must be equal in length")

    # Build csd
    c1 = np.ones(dni.shape, dtype=float)
    c2 = np.ones(dni.shape, dtype=float)

    # DNI for Zhao method is on 10 min basis, and so the 1-min inputs must be
    # sampled to 10 mins.
    # Take mean 10-min values. Potential issue with non-continuous data...
    hdni = _nanmean_reshape_10min(dni)
    hdnics = _nanmean_reshape_10min(dnics)

    # Test 1: Clearness index
    # From [1]; The clear-sky DNI is identified by the combine of two criterion
    # [Larraneta 2017]. The first criterion depends on the clearness index.
    # In this study, Dcs < 0.25 is considered as a clear-sky value.
    Dcs = np.abs((hdnics - hdni) / hdnics)
    c1[:len(Dcs), 0][Dcs < 0.25] = 0

    # Test 2: line length criterion
    # The second criterion is the length of the DNI curve.
    lcs = np.concatenate(([np.nan], np.sqrt((hdnics[1:] - hdnics[:-1]) ** 2)))
    lm = np.concatenate(([np.nan], np.sqrt((hdni[1:] - hdni[:-1]) ** 2)))
    lm[(lm == 0) & (hdni > 10)] = 0.001

    ldcs = np.abs((lcs - lm) / lm)
    c2[:len(ldcs), 0][ldcs < 3] = 0

    # Apply the criteria
    # Hourly criteria
    criteria = c1 + c2

    # Now scale back down to 1-min values
    x = np.arange(1, len(criteria) + 1, dtype=float)
    xx = np.linspace(1, len(hdni), len(hdni) * 10)

    criteria_flat = criteria.reshape(-1)
    criteria_1min = np.interp(xx, x, criteria_flat)

    # Add nans to the final part that could not be processed because it was incomplete
    criteria_1min = np.concatenate(
        [criteria_1min, np.full(len(dni) - len(criteria_1min), np.nan)]
    ).reshape(-1, 1)

    # Instantiate the csd
    csd = np.zeros(dni.shape, dtype=float)
    csd[criteria_1min != 0] = 1

    # Figure example
    if plot_figure is not None:
        plt.figure(num="Zhao CSD example", facecolor="w")
        plt.plot(dni, label="DNI")
        CSD = dni.astype(float).copy()
        CSD[csd == 1] = np.nan
        plt.plot(CSD, linewidth=2, color="k", label="CSD")
        plt.ylabel("Irradiance [Wm$^{-2}$]")
        plt.xticks([])
        plt.xlabel("Time [min]")
        plt.legend()
        plt.show()

    return csd