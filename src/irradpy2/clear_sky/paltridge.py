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


def paltridge(sza: np.ndarray):
    """
    Paltridge 1976 clear-sky model.

    References
    ----------
    Paltridge, G. W., & Platt, C. M. R. (1976).
    Radiative Processes in Meteorology and Climatology.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using
    routine hourly global irradiance measurements in Romania.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]

    Returns
    -------
    [EghPALTRIDGE, EbnPALTRIDGE, EdhPALTRIDGE] : list of np.ndarray
        EghPALTRIDGE : Global horizontal irradiance [W m-2]
        EbnPALTRIDGE : Direct normal irradiance [W m-2]
        EdhPALTRIDGE : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    sza_deg = sza / np.pi * 180
    lower = 0

    # direct normal irradiance
    EbnPALTRIDGE = 1000 * (1 - np.exp(-0.06 * (90 - sza_deg)))
    EbnPALTRIDGE[EbnPALTRIDGE < lower] = lower

    # diffuse horizontal irradiance
    EdhPALTRIDGE = 5 + 96 * (1 - np.exp(-0.05 * (90 - sza_deg)))
    EdhPALTRIDGE[EdhPALTRIDGE < lower] = lower

    # global horizontal irradiance
    EghPALTRIDGE = EbnPALTRIDGE * np.cos(sza) + EdhPALTRIDGE
    EghPALTRIDGE[EghPALTRIDGE < lower] = lower

    return [EghPALTRIDGE, EbnPALTRIDGE, EdhPALTRIDGE]