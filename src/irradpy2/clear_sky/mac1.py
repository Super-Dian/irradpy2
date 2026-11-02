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


def mac1(sza: np.ndarray, datetime, pressure: np.ndarray, wv: np.ndarray, albedo: np.ndarray):
    """
    MAC1 1982 clear-sky model.

    References
    ----------
    Davies, J. A., & McKay, D. C. (1982).
    Estimating solar irradiance and components.

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
    albedo : np.ndarray
        Ground albedo

    Returns
    -------
    [EghMAC1, EbnMAC1, EdhMAC1] : list of np.ndarray
        EghMAC1 : Global horizontal irradiance [W m-2]
        EbnMAC1 : Direct normal irradiance [W m-2]
        EdhMAC1 : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    wv = np.asarray(wv, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    amr = 35 / (1224 * np.cos(sza) ** 2 + 1) ** 0.5
    am = amr * pressure / 1013.25

    w1 = 10 * wv
    omeg = 0.98
    rhog = albedo

    amexp = 0.0279286 * np.log(am) - 0.806955
    X = 8.688237 * am ** amexp
    TrR = X / (1 + X)

    # ozone transmittance
    x = amr * 3.5
    aov = 0.002118 * x / (1 + 0.0042 * x + 0.00000323 * x * x)
    aou = 0.1082 * x / (1 + 13.86 * x) ** 0.805 + 0.00658 * x / (1 + (10.36 * x) ** 3)
    Tro = 1 - aov - aou

    # water vapour
    wp = w1 * amr * (pressure / 1013.25) ** 0.75
    aw = 0.29 * wp / ((1 + 14.15 * wp) ** 0.635 + 0.5925 * wp)

    Tabs = TrR * Tro - aw

    Tra = 0.95 ** am
    Trab = 0.95 ** 1.66

    # direct normal irradiance
    EbnMAC1 = Eext * Tabs * Tra

    # diffuse horizontal irradiance
    Eb = EbnMAC1 * np.cos(sza)
    Fa = 0.93 - 0.21 * np.log(amr)
    Fab = 0.93 - 0.21 * np.log(1.66)
    rhos = 0.0685 + (1 - Trab) * omeg * (1 - Fab)
    Ed0 = Eext * np.cos(sza) * (0.5 * Tro * Trab * (1 - TrR) + omeg * Fa * Tabs * (1 - Tra))
    Edb = rhog * rhos * (Eb + Ed0) / (1 - rhog * rhos)
    EdhMAC1 = Ed0 + Edb

    # global horizontal irradiance
    EghMAC1 = EbnMAC1 * np.cos(sza) + EdhMAC1

    # Quality control
    lower = 0
    EbnMAC1[EbnMAC1 < lower] = lower
    EdhMAC1[EdhMAC1 < lower] = lower
    EghMAC1[EghMAC1 < lower] = lower

    return [EghMAC1, EbnMAC1, EdhMAC1]