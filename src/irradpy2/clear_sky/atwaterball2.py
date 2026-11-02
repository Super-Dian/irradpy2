import numpy as np
import pandas as pd


def datetime_convert(datetime):
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)

    return Dayth


def atwaterball2(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, wv: np.ndarray):
    """
    Original Atwater and Ball_1 clear-sky model.

    References
    ----------
    Atwater, M. A., & Ball, J. T. (1981). A surface solar radiation model for cloudy atmospheres.
    Monthly Weather Review, 109(4), 878-888.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth automatically.
    press : np.ndarray
        Local barometric pressure [mb]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    Eghatwater1 : np.ndarray
        Global horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth= datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    press = np.asarray(press, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    Eext = Esc * (1 + 0.033 * np.cos(2 * np.pi * Dayth / 365))

    # Air Mass for Atwater
    am = 35 / (1224 * (np.cos(sza)) ** 2 + 1) ** 0.5

    # the transmittance after Rayleigh scattering TR and absorptance of permanent gases Tg
    presskpa = press / 10
    TRTg = 1.021 - 0.0824 * ((949 * presskpa * (10 ** -5) + 0.051) * am) ** 0.5

    # water vapor transmittance
    uW = wv * am
    TW = 1 - 2.9 * uW / (5.925 * uW + (1 + 141.5 * uW) ** 0.635)

    # the transmittance function for multiple layers of clouds
    Tc = 1

    # the transmittance after absorptance and reflectance of aerosols
    Tp = 1

    # atmospheric reflectance
    rs = 0.0685

    # ground albedo
    rg = -0.0139 + 0.0467 * np.tan(sza)
    rg[rg > 1] = 1
    rg[rg < 0.03] = 0.03

    # global horizontal irradiance
    Eghatwater1 = Eext * np.cos(sza) * TRTg * TW * Tc * Tp / (1 - rg * rs)

    # Quality control
    lower = 0
    Eghatwater1[Eghatwater1 < lower] = 0

    return Eghatwater1