import numpy as np
import pandas as pd


def datetime_convert(datetime):
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def dogniaux(sza: np.ndarray, datetime: np.ndarray, pressure: np.ndarray, decli: np.ndarray, TL2_series: list[np.ndarray]):
    """
    Dogniaux 1976 clear-sky model.

    References
    ----------
    Dogniaux, B. (1976).
    Programme de calcul des eclairements solaires energetiques et lumineux
    de surfaces orientees et inclinees.

    Dogniaux, B. (1970).
    Variations qualitatives et quantitatives des composantes du rayonnement
    solaire sur une surface horizontale par ciel serein en fonction du
    trouble atmospherique.

    Dogniaux, B. (1975).
    Variations geographiques et climatiques des expositions energetiques
    solaires sur des surfaces receptrices horizontales et verticales.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using
    routine hourly global irradiance measurements in Romania.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    pressure : np.ndarray
        Local barometric pressure [mb]
    decli : np.ndarray
        Solar declination angle [degrees]
    TL2_series : list[np.ndarray]
        A list of TL2 series. Input as many TL2 series as needed.
        Each TL2 represents Linke turbidity at air mass = 2.

    Returns
    -------
    [EghDOGNIAUX, EbnDOGNIAUX, EdhDOGNIAUX] : list
        EghDOGNIAUX : list of global horizontal irradiance series
        EbnDOGNIAUX : list of direct normal irradiance series
        EdhDOGNIAUX : list of diffuse horizontal irradiance series
        The order of each list follows the input order of TL2_series.
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    decli = np.asarray(decli, dtype=float)
    TL2_series = [np.asarray(TL2, dtype=float) for TL2 in TL2_series]

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # solar angles
    degrad = 0.0174532925
    sza_deg = sza / np.pi * 180
    h = 90 - sza_deg
    hr = h * degrad
    sh = np.cos(sza)
    s2h = np.sin(2 * hr)
    s3h = np.sin(3 * hr)
    ch = np.cos(hr)
    c2h = np.cos(2 * hr)
    c3h = np.cos(3 * hr)

    # air mass Kasten's
    amk = 1 / (np.cos(sza) + 0.15 * (93.885 - sza_deg) ** (-1.253))
    presc = pressure / 1013.25
    am = amk * presc

    tCDA = 1.4899 - 2.1099 * ch + 0.6322 * c2h + 0.0252 * c3h - 1.0022 * sh + 1.0077 * s2h - 0.2606 * s3h

    EghDOGNIAUX = []
    EbnDOGNIAUX = []
    EdhDOGNIAUX = []
    lower = 0

    for TL2 in TL2_series:
        # direct normal irradiance
        Ebn_i = Eext * np.exp(-tCDA * am * TL2 * 0.8662)
        Ebn_i[Ebn_i < lower] = lower

        # global horizontal irradiance
        XK = (0.051 + 0.018 * am) * TL2 * 0.8662 - 0.0016 * decli - 0.0895
        Eb = Ebn_i * np.cos(sza)
        Egh_i = Eb / (1 - XK)
        Egh_i[Egh_i < lower] = lower

        # diffuse horizontal irradiance
        Edh_i = Egh_i - Eb
        Edh_i[Edh_i < lower] = lower

        EghDOGNIAUX.append(Egh_i)
        EbnDOGNIAUX.append(Ebn_i)
        EdhDOGNIAUX.append(Edh_i)

    return [EghDOGNIAUX, EbnDOGNIAUX, EdhDOGNIAUX]