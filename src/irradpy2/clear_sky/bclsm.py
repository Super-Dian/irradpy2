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


def bclsm(sza: np.ndarray, datetime, wv: np.ndarray):
    """
    BCLSM 1979 clear-sky model.

    References
    ----------
    Barbaro, S. (1979).
    An atmospheric model for computing direct and diffuse solar radiation.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models
    using routine hourly global irradiance measurements in Romania.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    wv : np.ndarray
        Total precipitable water vapour [cm]

    Returns
    -------
    [EghBCLSM, EbnBCLSM, EdhBCLSM] : list of np.ndarray
        EghBCLSM : Global horizontal irradiance [W m-2]
        EbnBCLSM : Direct normal irradiance [W m-2]
        EdhBCLSM : Diffuse horizontal irradiance [W m-2]

    Notes
    -----
    Diffuse horizontal irradiance is calculated from
    EghBCLSM - EbnBCLSM * cos(sza).
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1369
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass Kasten's
    amk = 1 / (np.cos(sza) + 0.15 * (93.885 - sza / np.pi * 180) ** (-1.253))

    wm = 10 * wv * amk
    d = 200
    a1 = -0.13491
    b1 = -0.00428
    a2 = 0.13708
    b2 = 0.00261
    a3 = 3.68e-5
    b3 = 1.131e-4

    lower = 0

    # direct normal irradiance
    EbnBCLSM = Eext * np.exp(a1 + b1 * wv - a3 * (d - 400)) * np.exp(-amk * (a2 + b2 * wv + b3 * (d - 400)))
    EbnBCLSM[EbnBCLSM < lower] = lower

    # global horizontal irradiance
    Ewz0 = Eext * (0.938 * np.exp(-0.0154 * wm))
    Ewz1 = 0.1 * (41840 / 60) * (0.004 * wm ** 2.1 - 1.1086e-5 * wm ** 3 + 121.948 * (1 + wm) / (1 + 10 * wm ** 2))
    Ewz = Ewz0 + Ewz1
    Ed = np.maximum(0, 0.5 * (Ewz - EbnBCLSM) * np.cos(sza) ** 1.33333)
    EghBCLSM = (Ed + EbnBCLSM * np.cos(sza)) / 0.96

    # diffuse horizontal irradiance
    EdhBCLSM = EghBCLSM - EbnBCLSM * np.cos(sza)
    EdhBCLSM[EdhBCLSM < lower] = lower

    return [EghBCLSM, EbnBCLSM, EdhBCLSM]