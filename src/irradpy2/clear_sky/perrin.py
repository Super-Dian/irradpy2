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


def perrin(sza: np.ndarray, datetime, press: np.ndarray, ang_beta: np.ndarray, wv: np.ndarray, ozone: np.ndarray):
    """
    Original Perrin clear-sky model.

    References
    ----------
    de Brichambaut, C. P. (1975). Estimation des Ressources Energétiques en France.
    Cahiers de l’AFEDES, (1).
    Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
    in Romania. Renewable Energy, 55, 85-103.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    press : np.ndarray
        Local barometric pressure [mb]
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]

    Returns
    -------
    [EghPerrin, EbnPerrin, EdhPerrin] : list of np.ndarray
        EghPerrin : Global horizontal irradiance [W m-2]
        EbnPerrin : Direct normal irradiance [W m-2]
        EdhPerrin : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ozone = np.asarray(ozone, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    ama = am * press / 1013.25

    # atmospheric transmittance terms
    Trr = np.exp(-0.031411 - 0.064331 * ama)

    ako3 = am * ozone
    abso3 = 0.015 + 0.024 * ako3
    Tro = 1 - abso3

    akw = am * wv
    xw = np.zeros_like(wv, dtype=float)
    idx_xw = akw > 0
    xw[idx_xw] = np.log(akw[idx_xw])
    xw2 = xw * xw
    absw = 0.1 + 0.03 * xw + 0.002 * xw2
    Trw = 1 - absw

    akwg = ama * wv
    xwg = np.zeros_like(wv, dtype=float)
    idx_xwg = akwg > 0
    xwg[idx_xwg] = np.log(akwg[idx_xwg])
    absg = 0.013 - 0.0015 * xwg
    Trg = 1 - absg

    Tra = np.exp(-1.4327 * am * ang_beta)

    # direct normal irradiance
    EbnPerrin = Eext * Trr * Tra * (1 - abso3 - absw - absg)

    # global horizontal irradiance
    tCDA = 1 / (9.4 + 0.9 * ama)
    TL = -(np.log(EbnPerrin / Eext)) / (tCDA * ama)
    EghPerrin = (1270 - 56 * TL) * (np.cos(sza) ** ((TL + 36) / 33))

    # diffuse horizontal irradiance
    EdhPerrin = EghPerrin - EbnPerrin * np.cos(sza)

    # strange limit in Badescu
    idx_egh = EdhPerrin < 0
    EghPerrin[idx_egh] = EbnPerrin[idx_egh] * np.cos(sza)[idx_egh]

    # Quality control
    lower = 0
    EbnPerrin[EbnPerrin < lower] = 0
    EdhPerrin[EdhPerrin < lower] = 0
    EghPerrin[EghPerrin < lower] = 0

    return [EghPerrin, EbnPerrin, EdhPerrin]