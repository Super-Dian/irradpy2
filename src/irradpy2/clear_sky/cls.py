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


def cls(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, wv: np.ndarray, albedo: np.ndarray):
    """
    Original CLS clear-sky model.

    References
    ----------
    Suckling, P. W., & Hay, J. E. (1976). Modelling direct, diffuse, and total solar radiation for cloudless days.
    Atmosphere, 14(4), 298-308.
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
    [EghCLS, EbnCLS, EdhCLS] : list of np.ndarray
        EghCLS : Global horizontal irradiance [W m-2]
        EbnCLS : Direct normal irradiance [W m-2]
        EdhCLS : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    wv = np.asarray(wv, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    amk = 1 / (np.cos(sza) + 0.15 * (((np.pi / 2 - sza) / np.pi * 180) + 3.885) ** (-1.253))
    am = amk * press / 1013.25
    am2 = am * am
    am3 = am2 * am
    am4 = am3 * am

    # water vapour terms
    wm = wv * am
    wmb = wv * 1.66

    # atmospheric transmittances
    TrR = 0.972 - 0.08262 * am + 0.00933 * am2 - 0.00095 * am3 + 4.37e-5 * am4
    Trrb = 0.85655
    Tws = 1 - 0.0225 * wm
    Twsb = 1 - 0.0225 * wmb
    Twa = 1 - 0.077 * (wm ** 0.3)
    Twab = 1 - 0.077 * (wmb ** 0.3)
    Tda = 0.975 ** am
    Tdab = 0.975 ** 1.66
    Tds = Tda
    Tdsb = Tdab

    # direct normal irradiance
    EbnCLS = Eext * TrR * Tda * Tds * Twa * Tws

    # diffuse horizontal irradiance
    Eds = 0.6 * Eext * np.cos(sza) * Twa * Tda * (1 - Tws * Tds * TrR)
    Es = EbnCLS * np.cos(sza) + Eds
    Edb = albedo * Es * 0.4 * Twab * Tdab * (1 - Twsb * Tdsb * Trrb)
    EdhCLS = Eds + Edb

    # global horizontal irradiance
    EghCLS = Es + Edb

    # Quality control
    lower = 0
    EbnCLS[EbnCLS < lower] = 0
    EdhCLS[EdhCLS < lower] = 0
    EghCLS[EghCLS < lower] = 0

    return [EghCLS, EbnCLS, EdhCLS]