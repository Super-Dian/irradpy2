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


def furich(sza: np.ndarray, datetime: np.ndarray, altitude: np.ndarray):
    """
    Original Fu and Rich clear-sky model.

    References
    ----------
    Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
    Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
    Solar Energy, 86(8), 2145-2169.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    altitude : np.ndarray
        Location altitude [m]

    Returns
    -------
    [Eghfurich, Ebnfurich, Edhfurich] : list of np.ndarray
        Eghfurich : Global horizontal irradiance [W m-2]
        Ebnfurich : Direct normal irradiance [W m-2]
        Edhfurich : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    altitude = np.asarray(altitude, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # the air mass corrected for elevation
    mf = np.exp(-0.000118 * altitude - 1.638e-9 * altitude ** 2) / np.cos(sza)

    # a bulk atmospheric transmittance
    Tb = 0.5  # recommended value in Gueymard 2012

    # direct beam irradiance
    Ebnfurich = Eext * (Tb ** mf)

    # diffuse fraction of global normal irradiance
    P = 0.3  # recommended value in Gueymard 2012

    # diffuse horizontal irradiance
    Edhfurich = Ebnfurich * (P / (1 - P)) * np.cos(sza)

    # global horizontal irradiance
    Eghfurich = Ebnfurich * np.cos(sza) + Edhfurich

    # Quality control
    lower = 0
    Ebnfurich[Ebnfurich < lower] = 0
    Edhfurich[Edhfurich < lower] = 0
    Eghfurich[Eghfurich < lower] = 0

    return [Eghfurich, Ebnfurich, Edhfurich]