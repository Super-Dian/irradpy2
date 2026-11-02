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


def tj(sza: np.ndarray, datetime):
    """
    TJ 1957 clear-sky model.

    References
    ----------
    Threlkeld, J. L., & Jordan, R. C. (1957).
    Direct solar radiation available on clear days.
    Heat., Piping Air Cond., 29(12).

    Masters, G. M. (2013).
    Renewable and efficient electric power systems.
    John Wiley & Sons.

    Inputs
    ------
    sza : array-like
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input

    Outputs
    -------
    [EbnTJ, EdhTJ, EghTJ]
        EbnTJ : Direct normal irradiance [W m-2]
        EdhTJ : Diffuse horizontal irradiance [W m-2]
        EghTJ : Global horizontal irradiance [W m-2]

    Notes
    -----
    This function is translated directly from the original R code.
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)

    # air mass
    mA = 1 / np.cos(sza)

    # coefficients
    A = 1160 + 75 * np.sin(360 * (Dayth - 275) / 365 * np.pi / 180)   # degrees to radians
    k = 0.174 + 0.035 * np.sin(360 * (Dayth - 100) / 365 * np.pi / 180)   # degrees to radians
    C = 0.095 + 0.04 * np.sin(360 * (Dayth - 100) / 365 * np.pi / 180)    # degrees to radians

    # direct normal irradiance
    EbnTJ = A * np.exp(-k * mA)

    # diffuse horizontal irradiance
    EdhTJ = C * EbnTJ

    # global horizontal irradiance
    EghTJ = EbnTJ * np.cos(sza) + EdhTJ

    # Quality control
    lower = 0
    EbnTJ[EbnTJ < lower] = 0
    EdhTJ[EdhTJ < lower] = 0
    EghTJ[EghTJ < lower] = 0

    return [EghTJ, EbnTJ, EdhTJ]