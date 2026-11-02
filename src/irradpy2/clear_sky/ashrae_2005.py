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


def ashrae_2005(sza: np.ndarray, datetime):
    """
    ASHRAE 2005 clear-sky model.

    References
    ----------
    ASHRAE. (2005).
    ASHRAE Handbook—Fundamentals.
    American Society of Heating, Refrigerating and Air-Conditioning
    Engineers, Atlanta, GA, USA.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models
    using routine hourly global irradiance measurements in Romania.

    Inputs
    ------
    sza : array-like
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input. It is used to obtain the month automatically.

    Outputs
    -------
    [EghASHRAE, EbnASHRAE, EdhASHRAE]
        EghASHRAE : Global horizontal irradiance [W m-2]
        EbnASHRAE : Direct normal irradiance [W m-2]
        EdhASHRAE : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Month = np.asarray(time.month, dtype=float).reshape(original_shape)

    # coefficients of A, B and C from January to December
    A = np.array([1202, 1187, 1164, 1130, 1106, 1092, 1093, 1107, 1136, 1166, 1190, 1204], dtype=float)
    B = np.array([0.141, 0.142, 0.149, 0.164, 0.177, 0.185, 0.186, 0.182, 0.165, 0.152, 0.144, 0.141], dtype=float)
    C = np.array([0.103, 0.104, 0.109, 0.120, 0.130, 0.137, 0.138, 0.134, 0.121, 0.111, 0.106, 0.103], dtype=float)

    lower = 0

    # direct normal irradiance
    EbnASHRAE = A[0] * np.exp(-B[0] / np.cos(sza))
    EbnASHRAE[Month == 2] = A[1] * np.exp(-B[1] / np.cos(sza[Month == 2]))
    EbnASHRAE[Month == 3] = A[2] * np.exp(-B[2] / np.cos(sza[Month == 3]))
    EbnASHRAE[Month == 4] = A[3] * np.exp(-B[3] / np.cos(sza[Month == 4]))
    EbnASHRAE[Month == 5] = A[4] * np.exp(-B[4] / np.cos(sza[Month == 5]))
    EbnASHRAE[Month == 6] = A[5] * np.exp(-B[5] / np.cos(sza[Month == 6]))
    EbnASHRAE[Month == 7] = A[6] * np.exp(-B[6] / np.cos(sza[Month == 7]))
    EbnASHRAE[Month == 8] = A[7] * np.exp(-B[7] / np.cos(sza[Month == 8]))
    EbnASHRAE[Month == 9] = A[8] * np.exp(-B[8] / np.cos(sza[Month == 9]))
    EbnASHRAE[Month == 10] = A[9] * np.exp(-B[9] / np.cos(sza[Month == 10]))
    EbnASHRAE[Month == 11] = A[10] * np.exp(-B[10] / np.cos(sza[Month == 11]))
    EbnASHRAE[Month == 12] = A[11] * np.exp(-B[11] / np.cos(sza[Month == 12]))
    EbnASHRAE[EbnASHRAE < lower] = lower

    # diffuse horizontal irradiance
    EdhASHRAE = C[0] * EbnASHRAE
    EdhASHRAE[Month == 2] = C[1] * EbnASHRAE[Month == 2]
    EdhASHRAE[Month == 3] = C[2] * EbnASHRAE[Month == 3]
    EdhASHRAE[Month == 4] = C[3] * EbnASHRAE[Month == 4]
    EdhASHRAE[Month == 5] = C[4] * EbnASHRAE[Month == 5]
    EdhASHRAE[Month == 6] = C[5] * EbnASHRAE[Month == 6]
    EdhASHRAE[Month == 7] = C[6] * EbnASHRAE[Month == 7]
    EdhASHRAE[Month == 8] = C[7] * EbnASHRAE[Month == 8]
    EdhASHRAE[Month == 9] = C[8] * EbnASHRAE[Month == 9]
    EdhASHRAE[Month == 10] = C[9] * EbnASHRAE[Month == 10]
    EdhASHRAE[Month == 11] = C[10] * EbnASHRAE[Month == 11]
    EdhASHRAE[Month == 12] = C[11] * EbnASHRAE[Month == 12]

    # global horizontal irradiance
    EghASHRAE = EbnASHRAE * np.cos(sza) + EdhASHRAE
    EghASHRAE[EghASHRAE < lower] = lower

    return [EghASHRAE, EbnASHRAE, EdhASHRAE]