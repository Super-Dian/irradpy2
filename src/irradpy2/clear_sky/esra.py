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


def esra(sza: np.ndarray, datetime: np.ndarray, altitude: np.ndarray, TL2_series: list[np.ndarray]):
    """
    Original ESRA clear-sky model.

    References
    ----------
    Rigollier, C., Bauer, O., & Wald, L. (2000). On the clear sky model of the ESRA—European Solar Radiation Atlas—with respect to the Heliosat method.
    Solar Energy, 68(1), 33-48.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    altitude : np.ndarray
        Location altitude [m]
    TL2_series : list[np.ndarray]
        A list of TL2 series. Input as many TL2 series as needed.
        Each TL2 represents Linke turbidity at air mass = 2.

    Returns
    -------
    [EghEsra, EbnEsra, EdhEsra] : list
        EghEsra : list of global horizontal irradiance series
        EbnEsra : list of direct normal irradiance series
        EdhEsra : list of diffuse horizontal irradiance series
        The order of each list follows the input order of TL2_series.
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    altitude = np.asarray(altitude, dtype=float)
    TL2_series = [np.asarray(TL2, dtype=float) for TL2 in TL2_series]

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    alpha = np.pi / 2 - sza
    alphatrue = alpha + 0.061359 * (0.1594 + 1.123 * alpha + 0.065656 * alpha ** 2) / (1 + 28.9344 * alpha + 277.3971 * alpha ** 2)
    ame = np.exp(-altitude / 8434.5) / (np.sin(alphatrue) + 0.50572 * ((alphatrue / np.pi * 180 + 6.07995) ** -1.6364))

    # Rayleigh Optical Thickness
    rot = (6.6296 + 1.7513 * ame - 0.1202 * ame ** 2 + 0.0065 * ame ** 3 - 0.00013 * ame ** 4) ** -1
    rot[ame > 20] = (10.4 + 0.718 * ame[ame > 20]) ** -1

    EghEsra = []
    EbnEsra = []
    EdhEsra = []
    lower = 0

    for TL2 in TL2_series:
        # direct beam irradiance
        Ebn_i = np.maximum(0, Eext * np.exp(-0.8662 * TL2 * ame * rot))
        Ebn_i[Ebn_i < lower] = 0

        # diffuse horizontal irradiance
        TRD = -1.5843e-2 + 3.0543e-2 * TL2 + 3.797e-4 * TL2 ** 2
        a01 = 2.6463e-1 - 6.1581e-2 * TL2 + 3.1408e-3 * TL2 ** 2
        a11 = 2.0402 + 1.8945e-2 * TL2 - 1.1161e-2 * TL2 ** 2
        a21 = -1.3025 + 3.9231e-2 * TL2 + 8.5079e-3 * TL2 ** 2
        xiabiao = a01 * TRD < 2e-3
        a01[xiabiao] = (2e-3) / TRD[xiabiao]
        FD = a01 + a11 * np.sin(alpha) + a21 * np.sin(alpha) ** 2
        Edh_i = Eext * TRD * FD
        Edh_i[Edh_i < lower] = 0

        # global horizontal irradiance
        Egh_i = Ebn_i * np.cos(sza) + Edh_i
        Egh_i[Egh_i < lower] = 0

        EghEsra.append(Egh_i)
        EbnEsra.append(Ebn_i)
        EdhEsra.append(Edh_i)

    return [EghEsra, EbnEsra, EdhEsra]