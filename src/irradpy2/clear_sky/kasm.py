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



def kasm(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, wv: np.ndarray):
    """
    Original KASM clear-sky model.

    References
    ----------
    Kasten, F. (1984). Parametrisierung der Globalstahlung durch Bedeckungsgrad und Trubungsfaktor.
    Annalen der Meteorologie Neue, 20, 49-50.

    Badescu, V. (1997). Verification of some very simple clear and cloudy sky models to evaluate global solar irradiance.
    Solar Energy, 61(4), 251-264.

    Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements in Romania.
    Renewable Energy, 55, 85-103.

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

    Returns
    -------
    [EghKASM, EbnKASM, EdhKASM] : list of np.ndarray
        EghKASM : Global horizontal irradiance [W m-2]
        EbnKASM : Direct normal irradiance [W m-2]
        EdhKASM : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    amk = (np.cos(sza) + 0.15 * ((3.885 + (np.pi / 2 - sza) * 180 / np.pi) ** (-1.253))) ** (-1)
    am = amk * press / 1013.25
    am2 = am * am
    am3 = am2 * am
    am4 = am3 * am
    am5 = am4 * am

    w1 = 10 * wv
    TrR = 0.9768 - 0.0874 * am + 0.010607552 * am2 - 0.000846205 * am3 + 3.57246e-5 * am4 - 6.0176e-7 * am5

    x = amk * 3.5
    aov = 0.002118 * x / (1 + 0.0042 * x + 0.00000323 * x * x)
    aou = 0.1082 * x / ((1 + 13.86 * x) ** 0.805) + 0.00658 * x / (1 + (10.36 * x) ** 3)
    TroP = 1 - aov - aou

    wp = w1 * am
    aw = 0.29 * wp / ((1 + 14.15 * wp) ** 0.635 + 0.5925 * wp)
    Tra = 0.9 ** am

    # direct normal irradiance
    EbnKASM = Eext * (TrR * TroP - aw) * Tra

    # diffuse horizontal irradiance
    EdhKASM = Eext * np.cos(sza) * (TroP * (1 - TrR) * 0.5 + (TroP * TrR - aw) * (1 - Tra) * 0.75 * (0.93 - 0.21 * np.log(am)))

    # global horizontal irradiance
    EghKASM = EbnKASM * np.cos(sza) + EdhKASM

    # Quality control
    lower = 0
    EbnKASM[EbnKASM < lower] = 0
    EdhKASM[EdhKASM < lower] = 0
    EghKASM[EghKASM < lower] = 0

    return [EghKASM, EbnKASM, EdhKASM]