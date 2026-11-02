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


def ineichenperez(zsa: np.ndarray, datetime: np.ndarray, altitude: np.ndarray, TL2_series: list[np.ndarray]):
    """
    Original Ineichen & Perez clear-sky model.

    References
    ----------
    Ineichen, P., & Perez, R. (2002).
    A new airmass independent formulation for the Linke turbidity coefficient.
    Solar Energy, 73(3), 151-157.

    Parameters
    ----------
    zsa : np.ndarray
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
    Eghip : list[np.ndarray]
        Global horizontal irradiance series computed from the same model
        using each TL2 input in TL2_series, in the same order.
    """

    zsa = np.asarray(zsa, dtype=float)
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
    AM = 1 / (np.cos(zsa) + 0.50572 * ((6.07995 + (np.pi / 2 - zsa) * 180 / np.pi) ** (-1.6364)))

    # coefficients based on altitude
    fh1 = np.exp(-altitude / 8000)
    fh2 = np.exp(-altitude / 1250)
    cg1 = 5.09e-5 * altitude + 0.868
    cg2 = 3.92e-5 * altitude + 0.0387

    # global horizontal irradiance & QC
    Eghip = []
    lower = 0
    for TL2 in TL2_series:
        Egh_i = np.maximum(0, cg1 * Eext * np.cos(zsa) * np.exp(-cg2 * AM * (fh1 + fh2 * (TL2 - 1))))
        Egh_i[Egh_i < lower] = 0
        Eghip.append(Egh_i)

    return Eghip