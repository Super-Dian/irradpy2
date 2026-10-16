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


def cem(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, albedo: np.ndarray, broadbandaod: np.ndarray, wv: np.ndarray):
    """
    Original CEM clear-sky model.

    References
    ----------
    Atwater, M. A., & Ball, J. T. (1978). A numerical solar radiation model based on standard meteorological observations.
    Solar Energy, 21(3), 163-170.
    Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
    in Romania. Renewable Energy, 55, 85-103.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    press : np.ndarray
        Local barometric pressure [mb]
    albedo : np.ndarray
        Surface/ground albedo
    broadbandaod : np.ndarray
        Broadband aerosol optical depth [dimensionless]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghCEM, EbnCEM, EdhCEM] : list of np.ndarray
        EghCEM : Global horizontal irradiance [W m-2]
        EbnCEM : Direct normal irradiance [W m-2]
        EdhCEM : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    broadbandaod = np.asarray(broadbandaod, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # atmospheric terms
    am = 35 / (1224 * (np.cos(sza)) ** 2 + 1) ** 0.5
    amp = am * press / 1013.25
    p = press / 1013.25 * 101.325
    X1 = (am * (949 * p * 1e-5 + 0.051)) ** 0.5
    Trb = 1.041 - 0.16 * X1
    Trg = 1.021 - 0.0824 * X1
    Tg = 1
    To = 1
    absw = 0.077 * (am * wv) ** 0.3
    Trw = 1 - absw
    Tra = np.exp(-amp * broadbandaod)

    # direct normal irradiance
    EbnCEM = Eext * Tra * (Trb * Tg - absw)

    # global horizontal irradiance
    EghCEM = Eext * np.cos(sza) * Tra * (Trg * Tg - absw) / (1 - albedo * 0.0685)

    # diffuse horizontal irradiance
    EdhCEM = EghCEM - EbnCEM * np.cos(sza)

    # Quality control
    lower = 0
    EbnCEM[EbnCEM < lower] = 0
    EdhCEM[EdhCEM < lower] = 0
    EghCEM[EghCEM < lower] = 0

    return [EghCEM, EbnCEM, EdhCEM]