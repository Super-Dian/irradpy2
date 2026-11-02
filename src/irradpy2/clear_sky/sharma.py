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


def sharma(sza: np.ndarray,datetime):
    """
    Original Sharma clear-sky model.

    References
    ----------
    Sharma, M. R., & Pal, R. S. (1965).
    Interrelationships between total, direct, and diffuse solar radiation in the tropics.
    Solar Energy, 9(4), 183-192.

    Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M.,
    Dumitrescu, A., ... & Rada, C. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models
    using routine hourly global irradiance measurements in Romania.
    Renewable Energy, 55, 85-103.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input

    Returns
    -------
    [EghSharma, EbnSharma, EdhSharma] : list of np.ndarray
        EghSharma : Global horizontal irradiance [W m-2]
        EbnSharma : Direct normal irradiance [W m-2]
        EdhSharma : Diffuse horizontal irradiance [W m-2]
    """

    # -------------------------
    # Make safe local arrays
    # -------------------------
    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011+ 0.034221 * np.cos(B)+ 0.00128 * np.sin(B)+ 0.000719 * np.cos(2 * B)+ 0.000077 * np.sin(2 * B))

    # direct normal irradiance
    EbnSharma = 1.842 * (Eext / 2) * np.cos(sza) / (0.3135 + np.cos(sza))

    # global horizontal irradiance
    EghSharma = 4.5 * (Eext / (2 * 60)) + 1.071 * EbnSharma * np.cos(sza)

    # diffuse horizontal irradiance
    EdhSharma = EghSharma - EbnSharma * np.cos(sza)

    # Quality control
    lower = 0
    EbnSharma[EbnSharma < lower] = 0
    EdhSharma[EdhSharma < lower] = 0
    EghSharma[EghSharma < lower] = 0

    return [EghSharma, EbnSharma, EdhSharma]