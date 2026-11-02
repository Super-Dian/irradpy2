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


def heliosat(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, TL2_series: list[np.ndarray]):
    """
    Original Heliosat-1 clear-sky model.

    References
    ----------
    Hammer, A., Heinemann, D., Hoyer, C., Kuhlemann, R., Lorenz, E., Müller, R., & Beyer, H. G. (2003).
    Solar energy assessment using remote sensing technologies.
    Remote Sensing of Environment, 86(3), 423-432.

    Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
    Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
    Solar Energy, 86(8), 2145-2169.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    press : np.ndarray
        Local barometric pressure [mb]
    TL2_series : list[np.ndarray]
        A list of TL2 series. Input as many TL2 series as needed.
        Each TL2 represents Linke turbidity at air mass = 2.

    Returns
    -------
    [EghHeliosat1, EbnHeliosat1, EdhHeliosat1] : list
        EghHeliosat1 : list of global horizontal irradiance series
        EbnHeliosat1 : list of direct normal irradiance series
        EdhHeliosat1 : list of diffuse horizontal irradiance series
        The order of each list follows the input order of TL2_series.
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    TL2_series = [np.asarray(TL2, dtype=float) for TL2 in TL2_series]

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # absolute air mass
    AM = 1 / (np.cos(sza) + 0.50572 * ((6.07995 + (np.pi / 2 - sza) * 180 / np.pi) ** (-1.6364)))
    ama = AM * press / 1013.25

    # the Rayleigh optical thickness
    theta = 1 / (6.62960 + 1.7513 * ama - 0.1202 * ama ** 2 + 0.0065 * ama ** 3 - 0.00013 * ama ** 4)
    theta[ama > 20] = (10.4 + 0.718 * ama[ama > 20]) ** -1

    EghHeliosat1 = []
    EbnHeliosat1 = []
    EdhHeliosat1 = []
    lower = 0

    # calculate Ebn, Edh, Egh and quality control
    for TL2 in TL2_series:
        Ebn_i = np.maximum(0, Eext * np.exp(-ama * theta * TL2 * 0.8662))
        Ebn_i[Ebn_i < lower] = 0

        Edh_i = np.maximum(0, Eext * (0.0065 + (0.0646 * TL2 - 0.045) * np.cos(sza) - (0.0327 * TL2 - 0.014) * (np.cos(sza) ** 2)))
        Edh_i[Edh_i < lower] = 0

        Egh_i = Ebn_i * np.cos(sza) + Edh_i
        Egh_i[Egh_i < lower] = 0

        EghHeliosat1.append(Egh_i)
        EbnHeliosat1.append(Ebn_i)
        EdhHeliosat1.append(Edh_i)

    return [EghHeliosat1, EbnHeliosat1, EdhHeliosat1]