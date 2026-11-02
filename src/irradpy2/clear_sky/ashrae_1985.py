import numpy as np
import pandas as pd


def datetime_convert(datetime):
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Month = np.asarray(time.month, dtype=float).reshape(original_shape)

    return Month


def ashrae_1985(sza: np.ndarray, datetime: np.ndarray):
    """
    Original ASHRAE clear-sky model.

    References
    ----------
    Handbook, A. F. (1985).
    American society of heating, refrigerating and air-conditioning engineers.
    Inc.: Atlanta, GA, USA.

    El Mghouchi, Y., Ajzoul, T., Taoukil, D., & El Bouardi, A. (2016).
    The most suitable prediction model of the solar intensity, on horizontal plane,
    at various weather conditions in a specified location in Morocco.
    Renewable and Sustainable Energy Reviews, 54, 84-98.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Month automatically.

    Returns
    -------
    [EghASHRAE, EbnASHRAE, EdhASHRAE] : list of np.ndarray
        EghASHRAE : Global horizontal irradiance [W m-2]
        EbnASHRAE : Direct normal irradiance [W m-2]
        EdhASHRAE : Diffuse horizontal irradiance [W m-2]
    """

    # -------------------------
    # Make safe local arrays
    # -------------------------
    sza = np.asarray(sza, dtype=float)
    Month = datetime_convert(datetime)
    Month = np.asarray(Month)

    # coeff of A, B, C from Jan, Feb, ..., Dec
    A = np.array([1230, 1215, 1186, 1136, 1104, 1088, 1085, 1107, 1152, 1193, 1221, 1234], dtype=float)
    B = np.array([0.142, 0.144, 0.156, 0.180, 0.196, 0.205, 0.207, 0.201, 0.177, 0.160, 0.149, 0.142], dtype=float)
    C = np.array([0.058, 0.060, 0.071, 0.097, 0.121, 0.134, 0.136, 0.122, 0.092, 0.073, 0.063, 0.057], dtype=float)

    # direct normal irradiance
    EbnASHRAE = A[0] * np.exp(-B[0] / (np.cos(sza)))   # Jan
    EbnASHRAE[np.where(Month == 2)] = A[1] * np.exp(-B[1] / (np.cos(sza)[np.where(Month == 2)]))   # Feb
    EbnASHRAE[np.where(Month == 3)] = A[2] * np.exp(-B[2] / (np.cos(sza)[np.where(Month == 3)]))   # Mar
    EbnASHRAE[np.where(Month == 4)] = A[3] * np.exp(-B[3] / (np.cos(sza)[np.where(Month == 4)]))   # Apr
    EbnASHRAE[np.where(Month == 5)] = A[4] * np.exp(-B[4] / (np.cos(sza)[np.where(Month == 5)]))   # May
    EbnASHRAE[np.where(Month == 6)] = A[5] * np.exp(-B[5] / (np.cos(sza)[np.where(Month == 6)]))   # Jun
    EbnASHRAE[np.where(Month == 7)] = A[6] * np.exp(-B[6] / (np.cos(sza)[np.where(Month == 7)]))   # Jul
    EbnASHRAE[np.where(Month == 8)] = A[7] * np.exp(-B[7] / (np.cos(sza)[np.where(Month == 8)]))   # Aug
    EbnASHRAE[np.where(Month == 9)] = A[8] * np.exp(-B[8] / (np.cos(sza)[np.where(Month == 9)]))   # Sep
    EbnASHRAE[np.where(Month == 10)] = A[9] * np.exp(-B[9] / (np.cos(sza)[np.where(Month == 10)])) # Oct
    EbnASHRAE[np.where(Month == 11)] = A[10] * np.exp(-B[10] / (np.cos(sza)[np.where(Month == 11)])) # Nov
    EbnASHRAE[np.where(Month == 12)] = A[11] * np.exp(-B[11] / (np.cos(sza)[np.where(Month == 12)])) # Dec

    # diffuse horizontal irradiance
    EdhASHRAE = C[0] * EbnASHRAE   # Jan
    EdhASHRAE[np.where(Month == 2)] = C[1] * EbnASHRAE[np.where(Month == 2)]   # Feb
    EdhASHRAE[np.where(Month == 3)] = C[2] * EbnASHRAE[np.where(Month == 3)]   # Mar
    EdhASHRAE[np.where(Month == 4)] = C[3] * EbnASHRAE[np.where(Month == 4)]   # Apr
    EdhASHRAE[np.where(Month == 5)] = C[4] * EbnASHRAE[np.where(Month == 5)]   # May
    EdhASHRAE[np.where(Month == 6)] = C[5] * EbnASHRAE[np.where(Month == 6)]   # Jun
    EdhASHRAE[np.where(Month == 7)] = C[6] * EbnASHRAE[np.where(Month == 7)]   # Jul
    EdhASHRAE[np.where(Month == 8)] = C[7] * EbnASHRAE[np.where(Month == 8)]   # Aug
    EdhASHRAE[np.where(Month == 9)] = C[8] * EbnASHRAE[np.where(Month == 9)]   # Sep
    EdhASHRAE[np.where(Month == 10)] = C[9] * EbnASHRAE[np.where(Month == 10)] # Oct
    EdhASHRAE[np.where(Month == 11)] = C[10] * EbnASHRAE[np.where(Month == 11)] # Nov
    EdhASHRAE[np.where(Month == 12)] = C[11] * EbnASHRAE[np.where(Month == 12)] # Dec

    # global horizontal irradiance
    EghASHRAE = EbnASHRAE * np.cos(sza) + EdhASHRAE

    # Quality control
    lower = 0
    EbnASHRAE[EbnASHRAE < lower] = 0
    EdhASHRAE[EdhASHRAE < lower] = 0
    EghASHRAE[EghASHRAE < lower] = 0

    return [EghASHRAE, EbnASHRAE, EdhASHRAE]