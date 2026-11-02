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


def _nanmean_2d_axis0(x):
    """
    MATLAB-like nanmean(..., 1) for a 2D array.
    """
    return np.nanmean(x, axis=0)


def _interp1_linear(x, y, xi):
    """MATLAB-like interp1 linear interpolation with NaN propagation."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    out = np.full(xi.shape, np.nan, dtype=float)
    if x.size < 2:
        return out

    for index, value in np.ndenumerate(xi):
        exact = np.where(np.isclose(x, value, rtol=0.0, atol=1e-12))[0]
        if exact.size:
            out[index] = y[exact[0]]
            continue

        left = np.searchsorted(x, value) - 1
        if left < 0 or left >= x.size - 1:
            continue

        y0, y1 = y[left], y[left + 1]
        if np.isnan(y0) or np.isnan(y1):
            continue

        x0, x1 = x[left], x[left + 1]
        out[index] = y0 + (y1 - y0) * (value - x0) / (x1 - x0)

    return out


def Ineichen2016CSD(ghi, dni, dhi, zen, exth, aod550, plot_figure=None):
    """
    Ineichen 2016 Clear Sky Detection model.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance, column vector.
    dni : array-like
        Direct normal irradiance, column vector.
    dhi : array-like
        Diffuse horizontal irradiance, column vector.
    zen : array-like
        Zenith angle in degrees, column vector.
    exth : array-like
        Horizontal projection of extraterrestrial irradiance, column vector.
    aod550 : array-like
        Total aerosol extinction at 550 nm, column vector.
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
    dni = _ensure_column_vector(dni, "dni")
    dhi = _ensure_column_vector(dhi, "dhi")
    zen = _ensure_column_vector(zen, "zen")
    exth = _ensure_column_vector(exth, "exth")
    aod550 = _ensure_column_vector(aod550, "aod550")

    if len(np.unique([len(ghi), len(dni), len(dhi), len(zen), len(exth), len(aod550)])) != 1:
        raise ValueError("vars must be equal in length")

    # ------------------------------------------------------------------
    # Pre allocation
    # ------------------------------------------------------------------
    csd = np.zeros_like(ghi)

    # clearness index
    kt = ghi / exth

    # Optical air mass from [2]
    h = 90.0 - zen
    M = 1.0 / (np.sin(np.deg2rad(h)) + 0.15 * (h + 3.885) ** (-1.253))

    # Modified global or surface irradiance clearness index
    kt_prime = kt / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / M)) + 0.1)

    # ------------------------------------------------------------------
    # Criteria
    # ------------------------------------------------------------------

    # Closure equation:
    # GHIsum = cosd(zen).*dni + dif;
    GHIsum = np.cos(np.deg2rad(zen)) * dni + dhi
    csd[np.abs(GHIsum - ghi) > 50] = 1
    csd[np.abs(GHIsum - ghi) / ghi > 0.05] = 1

    # The global clearness index Kt of the measurements is lower than 0.82
    # MATLAB actually flags kt >= 0.82
    csd[kt >= 0.82] = 1

    # The modified global clearness Kt' of the measurements is higher than 0.65
    # MATLAB actually flags kt_prime <= 0.65
    csd[kt_prime <= 0.65] = 1

    # Stability of hourly kt_prime
    n_mod = len(kt_prime) % 60
    hr_kt_prime_mod = kt_prime[: len(kt_prime) - n_mod] if n_mod != 0 else kt_prime.copy()

    if len(hr_kt_prime_mod) > 0:
        hourly_kt_prime = _nanmean_2d_axis0(
            hr_kt_prime_mod.reshape(60, len(hr_kt_prime_mod) // 60, order="F")
        )

        dkt_prime_hr = np.concatenate((
            np.array([np.nan]),
            hourly_kt_prime[:-1] - hourly_kt_prime[1:]
        ))

        x = np.arange(1, len(dkt_prime_hr) + 1, dtype=float)
        xi = np.linspace(1, len(dkt_prime_hr), len(hr_kt_prime_mod))
        dkt_prime_min = _interp1_linear(x, dkt_prime_hr, xi).reshape(-1, 1)

        if n_mod != 0:
            dkt_prime_min = np.vstack((dkt_prime_min, np.full((n_mod, 1), np.nan)))
    else:
        dkt_prime_min = np.full_like(ghi, np.nan)

    csd[np.abs(dkt_prime_min) > 0.01] = 1

    # The broadband aerosol optical depth is lower than 0.5
    # MATLAB actually flags aod >= 0.5
    csd[aod550 >= 0.5] = 1

    # ------------------------------------------------------------------
    # Figure example
    # ------------------------------------------------------------------
    if plot_figure is not None:
        plt.figure(num="Ineichen 2016 CSD example", facecolor="w")
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