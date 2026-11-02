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


def ineichen_2008(sza: np.ndarray, datetime, press: np.ndarray, aod700: np.ndarray, wv: np.ndarray):
    """
    Original simplified Solis clear-sky model.

    References
    ----------
    Ineichen, P. (2008). A broadband simplified version of the Solis clear sky model.
    Solar Energy, 82(8), 758-762.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    press : np.ndarray
        Local barometric pressure [mb]
    aod700 : np.ndarray
        Aerosol optical depth at 700 nm [dimensionless]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [Eghsolis, Ebnsolis, Edhsolis] : list of np.ndarray
        Eghsolis : Global horizontal irradiance [W m-2]
        Ebnsolis : Direct normal irradiance [W m-2]
        Edhsolis : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    aod700 = np.asarray(aod700, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # Ineichen 2008 limits aod700 between 0 and 0.45
    aod700 = np.minimum(0.45, aod700)

    # modified solar constant
    Eext0 = 1.08 * wv ** 0.0051
    Eext1 = 0.97 * wv ** 0.032
    Eext2 = 0.12 * wv ** 0.56
    Eexte = Eext * (Eext2 * aod700 ** 2 + Eext1 * aod700 + Eext0 + 0.071 * np.log(press / 1013.25))

    # direct total optical depth
    tb1 = 1.82 + 0.056 * np.log(wv) + 0.0071 * (np.log(wv)) ** 2
    tb0 = 0.33 + 0.045 * np.log(wv) + 0.0096 * (np.log(wv)) ** 2
    tbp = 0.0089 * wv + 0.13
    tb = tb1 * aod700 + tb0 + tbp * np.log(press / 1013.25)
    b1 = 0.00925 * aod700 ** 2 + 0.0148 * aod700 - 0.0172
    b0 = -0.7565 * aod700 ** 2 + 0.5057 * aod700 + 0.4557
    b = b1 * np.log(wv) + b0

    # diffuse total optical depth
    td4 = 86 * wv - 13800
    td3 = -3.11 * wv + 79.4
    td2 = -0.23 * wv + 74.8
    td1 = 0.092 * wv - 8.86
    td0 = 0.0042 * wv + 3.12
    tdp = -0.83 * (1 + aod700) ** -17.2
    idx = aod700 >= 0.05
    td4[idx] = -0.21 * wv[idx] + 11.6
    td3[idx] = 0.27 * wv[idx] - 20.7
    td2[idx] = -0.134 * wv[idx] + 15.5
    td1[idx] = 0.0554 * wv[idx] - 5.71
    td0[idx] = 0.0057 * wv[idx] + 2.94
    tdp[idx] = -0.71 * (1 + aod700[idx]) ** -15
    td = td4 * aod700 ** 4 + td3 * aod700 ** 3 + td2 * aod700 ** 2 + td1 * aod700 + td0 + tdp * np.log(press / 1013.25)
    dp = 1 / (18 + 152 * aod700)
    d = -0.337 * aod700 ** 2 + 0.63 * aod700 + 0.116 + dp * np.log(press / 1013.25)

    # global total optical depth
    tg1 = 1.24 + 0.047 * np.log(wv) + 0.0061 * (np.log(wv)) ** 2
    tg0 = 0.27 + 0.043 * np.log(wv) + 0.0090 * (np.log(wv)) ** 2
    tgp = 0.0079 * wv + 0.1
    tg = tg1 * aod700 + tg0 + tgp * np.log(press / 1013.25)
    g = -0.0147 * np.log(wv) - 0.3079 * aod700 ** 2 + 0.2846 * aod700 + 0.3798

    # direct normal irradiance
    Ebnsolis = Eexte * np.exp(-tb / (np.cos(sza) ** b))

    # diffuse horizontal irradiance
    Edhsolis = Eexte * np.exp(-td / (np.cos(sza) ** d))

    # global horizontal irradiance
    Eghsolis = Eexte * np.exp(-tg / (np.cos(sza) ** g)) * np.cos(sza)

    # Quality control
    lower = 0
    Ebnsolis[Ebnsolis < lower] = 0
    Edhsolis[Edhsolis < lower] = 0
    Eghsolis[Eghsolis < lower] = 0

    return [Eghsolis, Ebnsolis, Edhsolis]