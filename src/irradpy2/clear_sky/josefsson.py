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


def josefsson(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, wv: np.ndarray, albedo: np.ndarray):
    """
    Original Josefsson clear-sky model.

    References
    ----------
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
    albedo : np.ndarray
        Surface/ground albedo

    Returns
    -------
    [EghJosefsson, EbnJosefsson, EdhJosefsson] : list of np.ndarray
        EghJosefsson : Global horizontal irradiance [W m-2]
        EbnJosefsson : Direct normal irradiance [W m-2]
        EdhJosefsson : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    wv = np.asarray(wv, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass and auxiliary variables
    amk = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    am = amk * press / 1013.25
    am2 = am * am
    am3 = am2 * am
    am4 = am3 * am
    am5 = am4 * am
    am66 = 1.66 * press / 1013.25

    h = (np.pi / 2 - sza) * 180 / np.pi
    w1 = 10 * wv
    omeg = 0.75
    wp = w1 * amk
    aw = 0.29 * wp / ((1 + 14.15 * wp) ** 0.635 + 0.5925 * wp)
    g = 0.5248 + 0.007981 * h
    h_xiabiao = np.where(h > 45)
    g[h_xiabiao] = 0.856 + 0.000734 * h[h_xiabiao]

    # atmospheric transmittances
    Tra = 0.95 ** amk
    Trab = 0.95 ** am66
    Traa = 1 - (1 - omeg) * (1 - Tra)
    Tras = 1 - omeg * (1 - Tra)
    TrO3 = 0.95545
    TrR = 0.9768 - 0.0874 * am + 0.010607552 * am2 - 0.000846205 * am3 + 3.57246e-5 * am4 - 6.0176e-7 * am5
    Tabs = TrO3 * Traa

    # direct normal irradiance
    EbnJosefsson = Eext * (Tabs * Tras * TrR - aw)
    Eb = EbnJosefsson * np.cos(sza)

    # diffuse horizontal irradiance
    Fa = 0.93 - 0.21 * np.log(amk)
    Fab = 0.93 - 0.21 * np.log(1.66)
    rhos = 0.0685 + (1 - Trab) * omeg * (1 - Fab)
    Ed0 = Eext * np.cos(sza) * (0.5 * Tabs * Tras * (1 - TrR) + g * (TrR * Tabs - aw) * (1 - Tras))
    Edb = albedo * rhos * (Eb + Ed0) / (1 - albedo * rhos)
    EdhJosefsson = Ed0 + Edb

    # global horizontal irradiance
    EghJosefsson = EbnJosefsson * np.cos(sza) + EdhJosefsson

    # Quality control
    lower = 0
    EbnJosefsson[EbnJosefsson < lower] = 0
    EdhJosefsson[EdhJosefsson < lower] = 0
    EghJosefsson[EghJosefsson < lower] = 0

    return [EghJosefsson, EbnJosefsson, EdhJosefsson]