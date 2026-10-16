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


def badescu(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, wv: np.ndarray, ozone: np.ndarray):
    """
    Original Badescu clear-sky model.

    References
    ----------
    Badescu, V. (2008). Use of sunshine number for solar irradiance time series generation.
    In Modeling Solar Radiation at the Earth’s Surface (pp. 327-355). Springer, Berlin, Heidelberg.
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
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]

    Returns
    -------
    [EghBadescu, EbnBadescu, EdhBadescu] : list of np.ndarray
        EghBadescu : Global horizontal irradiance [W m-2]
        EbnBadescu : Direct normal irradiance [W m-2]
        EdhBadescu : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
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

    # the transimissivity of the atmosphere due to Rayleigh scattering by molecules
    Tr = 0.9768 - 0.0874 * ama + 0.010607552 * ama ** 2 - 8.46205e-4 * ama ** 3 + 3.57246e-5 * ama ** 4 - 6.0176e-7 * ama ** 5

    # the transmissivity of the ozone layer
    XO = ama * (ozone * 10)
    AO = ((0.1082 * XO) / ((1 + 13.86 * XO) ** 0.805)) + ((0.00658 * XO) / (1 + (10.36 * XO) ** 3)) + (0.002118 * XO / (1 + 0.0042 * XO + 3.23e-6 * XO ** 2))
    TO = 1 - AO

    # the transmissivity of aerosols depends on the air mass
    Ta = 0.84 ** ama

    # the absorptivity by water vapour
    XW = ama * wv * 10
    AW = 0.29 * XW / (((1 + 14.15 * XW) ** 0.635) + 0.5925 * XW)

    # direct normal irradiance
    EbnBadescu = Eext * np.maximum((Tr * TO - AW), 0) * Ta

    # diffuse horizontal irradiance
    EdhBadescu = Eext * np.cos(sza) * (0.5 * TO * (1 - Tr) + 0.75 * (0.93 - 0.21 * np.log(ama)) * (TO * Tr - AW) * (1 - Ta))

    # global horizontal irradiance
    EghBadescu = EbnBadescu * np.cos(sza) + EdhBadescu

    # Quality control
    lower = 0
    EbnBadescu[EbnBadescu < lower] = 0
    EdhBadescu[EdhBadescu < lower] = 0
    EghBadescu[EghBadescu < lower] = 0

    return [EghBadescu, EbnBadescu, EdhBadescu]