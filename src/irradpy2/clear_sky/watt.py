import numpy as np
import pandas as pd


def datetime_convert(datetime):
    """Convert a datetime sequence to (day-of-year, year) arrays.

    Parameters
    ----------
    datetime : array-like
        Datetime sequence parsed with ``pandas.to_datetime``; invalid entries
        become NaT. The original array shape is preserved.

    Returns
    -------
    Dayth : numpy.ndarray of float
        Day of year (1-366), same shape as ``datetime``.
    Year : numpy.ndarray of float
        Calendar year, same shape as ``datetime``.
    """
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def watt(sza: np.ndarray, datetime, pressure: np.ndarray, wv: np.ndarray, ozone: np.ndarray, ang_beta: np.ndarray, ang_alpha: np.ndarray, albedo: np.ndarray):
    """
    Watt 1978 clear-sky model.

    References
    ----------
    Watt, D. (1978).
    On the Nature and Distribution of Solar Radiation.
    Report for the U.S. Department of Energy.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using
    routine hourly global irradiance measurements in Romania.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input. It is used to calculate Dayth and Year automatically.
    pressure : np.ndarray
        Local barometric pressure [mb]
    wv : np.ndarray
        Total precipitable water vapour [cm]
    ozone : np.ndarray
        Total ozone amount [atm-cm]
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ang_alpha : np.ndarray
        Angstrom exponent
    albedo : np.ndarray
        Ground albedo

    Returns
    -------
    [EghWATT, EbnWATT, EdhWATT] : list of np.ndarray
        EghWATT : Global horizontal irradiance [W m-2]
        EbnWATT : Direct normal irradiance [W m-2]
        EdhWATT : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B0 = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B0) + 0.00128 * np.sin(B0) + 0.000719 * np.cos(2 * B0) + 0.000077 * np.sin(2 * B0))

    h1 = np.array([0, 0, 15, 20], dtype=float)
    h2 = np.array([30, 1.5, 25, 40], dtype=float)

    Oz = 10 * ozone
    r = 6.4e3
    TU = 0.02
    B = ang_beta * 2 ** ang_alpha * 0.434294 - TU
    TL = 0.6 * (B - 0.01 * wv - 0.03)
    T = np.maximum(0, TL + TU)
    Ag = albedo

    am = np.empty(sza.shape + (4,), dtype=float)

    for i in range(4):
        X2 = r / h2[i]
        fz2 = ((X2 * np.cos(sza)) ** 2 + 2 * X2 + 1) ** 0.5 - X2 * np.cos(sza)
        am[..., i] = fz2

        if h1[i] <= 0:
            continue

        X1 = r / h1[i]
        fz1 = ((X1 * np.cos(sza)) ** 2 + 2 * X1 + 1) ** 0.5 - X1 * np.cos(sza)
        am[..., i] = (h2[i] * fz2 - h1[i] * fz1) / (h2[i] - h1[i])

    am1 = am[..., 0]
    am2 = am[..., 1]
    am3 = am[..., 2]
    am4 = am[..., 3]

    y1 = 0.045 * (pressure / 1013.25 * am1) ** 0.7
    y2 = 0.0071 + 0.001 * Oz * am4
    y3 = 0.0095 * wv * am2
    y4 = TL * am2 ** 0.7
    y5 = TU * am3
    Tw = 0.93 - 0.033 * np.log10(wv * am2)
    Tn = Tw * 10 ** (-y1 - y2 - y3 - y4 - y5)

    # direct normal irradiance
    EbnWATT = Eext * Tn

    # diffuse horizontal irradiance
    Acs = (0.93 - 0.033 * np.log10(wv)) * 10 ** (-0.006 * pressure / 1013.25 - 0.4 * T)
    As = Acs * (1 - 10 ** (-0.003 * pressure / 1013.25 - 0.01 * wv - 0.4 * T))
    Ed123 = Eext * 0.8 * As * (1 + Ag * As) * (1 + np.cos(sza)) ** 0.5
    Ed45 = Eext * np.cos(sza) * 0.5 * Acs * Ag * As
    Ed6 = Eext * np.cos(sza) * 0.5 * As
    EdhWATT = Ed123 + Ed45 + Ed6

    # global horizontal irradiance
    EghWATT = EdhWATT + EbnWATT * np.cos(sza)

    # Quality control
    lower = 0
    EbnWATT[EbnWATT < lower] = lower
    EdhWATT[EdhWATT < lower] = lower
    EghWATT[EghWATT < lower] = lower

    return [EghWATT, EbnWATT, EdhWATT]