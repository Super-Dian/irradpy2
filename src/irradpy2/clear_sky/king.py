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


def king(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, albedo: np.ndarray, wv: np.ndarray, ang_beta: np.ndarray):
    """
    Original King clear-sky model.

    References
    ----------
    King, R., & Buckius, R. O. (1979). Direct solar transmittance for a clear sky.
    Solar Energy, 22(3), 297-301.
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
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]
    ang_beta : np.ndarray
        Angstrom turbidity coefficient

    Returns
    -------
    [EghKing, EbnKing, EdhKing] : list of np.ndarray
        EghKing : Global horizontal irradiance [W m-2]
        EbnKing : Direct normal irradiance [W m-2]
        EdhKing : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    amk = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    am = amk * press / 1013.25
    amb = amk * ang_beta
    a1 = 0.5

    # direct transmittance terms
    Tscat = (5.228 / am) * (np.exp(-0.0002254 * am) - np.exp(-0.1409 * am)) * np.exp(-amb / (0.6489 ** 1.3)) + 0.00165 * am + 0.2022 * (1 - amb / (2.875 ** 1.3))
    Tabs = (0.1055 + 0.07053 * np.log10(amk * wv + 0.07854)) * np.exp(-amb / (1.519 ** 1.3))

    # direct normal irradiance
    EbnKing = Eext * (Tscat - Tabs)

    # diffuse horizontal irradiance
    taul = 0.8336 + 0.17257 * ang_beta - 0.64595 * np.exp(-7.4296 * ang_beta ** 1.5)
    Tml = np.exp(-am * taul)
    F1 = 2 * (1 + 2 * np.cos(sza) + (1 - 2 * np.cos(sza)) * Tml)
    F2 = (albedo - 1) * (a1 * taul - 4 * taul - 4) + 4 * albedo
    Tdif = 0.634 * (F1 / F2 - Tml)
    EdhKing = Eext * np.cos(sza) * Tdif

    # global horizontal irradiance
    EghKing = EbnKing * np.cos(sza) + EdhKing

    # Quality control
    lower = 0
    EbnKing[EbnKing < lower] = 0
    EdhKing[EdhKing < lower] = 0
    EghKing[EghKing < lower] = 0

    return [EghKing, EbnKing, EdhKing]