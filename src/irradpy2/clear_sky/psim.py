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


def psim(sza: np.ndarray, datetime, press: np.ndarray, albedo: np.ndarray, ang_beta: np.ndarray, wv: np.ndarray):
    """
    Original PSIM clear-sky model.

    References
    ----------
    Gueymard, C. (1993). Mathermatically integrable parameterization of clear-sky beam and global irradiances and its use in daily irradiation applications.
    Solar Energy, 50(5), 385-397.
    Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M., Dumitrescu, A., ... & Rada, C. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models using routine hourly global irradiance measurements
    in Romania. Renewable Energy, 55, 85-103.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    press : np.ndarray
        Local barometric pressure [mb]
    albedo : np.ndarray
        Surface/ground albedo
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghPSIM, EbnPSIM, EdhPSIM] : list of np.ndarray
        EghPSIM : Global horizontal irradiance [W m-2]
        EbnPSIM : Direct normal irradiance [W m-2]
        EdhPSIM : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1373
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # atmospheric and angular terms
    ATM = press / 1013.25
    Fbgw = np.exp(0.155 * (1 - wv ** 0.24))
    F2PB = 1 + (0.1594 - 0.226 * ang_beta) * (1 - ATM)
    F2PG = 1 + (0.0752 - 0.107 * ang_beta) * (1 - ATM)
    ROS = 0.061 + 0.072 * ang_beta ** 0.5
    F3RO = (1 - 0.2 * ROS) / (1 - albedo * ROS)
    S = np.maximum(np.cos(sza), 0.0285)
    S2 = S * S
    S3 = S2 * S
    S4 = S2 * S2
    S5 = S3 * S2
    X = np.log(1 + 10 * ang_beta)
    X2 = X * X
    X3 = X2 * X

    # coefficients for direct normal irradiance
    A0 = np.exp(-1.62364 - 6.8727 * X)
    A1 = 2.94298 + 5.23504 * X - 18.23861 * X2 + 11.1652 * X3
    A2 = -8.12158 - 15.8 * X + 69.2345 * X2 - 45.1637 * X3
    A3 = 12.5571 + 25.44 * X - 123.3933 * X2 + 83.1014 * X3
    A4 = -9.8044 - 20.3172 * X + 103.9906 * X2 - 71.3091 * X3
    A5 = 3.00487 + 6.3176 * X - 33.3891 * X2 + 23.1547 * X3

    # limit of ang_beta >= 0.175
    idx = ang_beta >= 0.175
    A0[idx] = 0
    A1[idx] = 4.41547 - 5.09266 * X[idx] + 1.471865 * X2[idx]
    A2[idx] = -18.45187 + 38.3584 * X[idx] - 22.7449 * X2[idx] + 4.3189 * X3[idx]
    A3[idx] = 31.2506 - 74.53835 * X[idx] + 48.355 * X2[idx] - 9.8657 * X3[idx]
    A4[idx] = -25.18762 + 64.3575 * X[idx] - 43.6586 * X2[idx] + 9.23151 * X3[idx]
    A5[idx] = 7.64179 - 20.41687 * X[idx] + 14.20502 * X2[idx] - 3.06053 * X3[idx]

    # direct normal irradiance
    EbnPSIM = Eext * Fbgw * F2PB * (A0 + A1 * S + A2 * S2 + A3 * S3 + A4 * S4 + A5 * S5)

    # global horizontal irradiance
    g0 = 0.006
    g1 = 0.387021 - 0.386246 * X + 0.09234 * X2
    g2 = 1.35369 + 1.533 * X - 1.07736 * X2 + 0.23728 * X3
    g3 = -1.59816 - 1.903774 * X + 1.631134 * X2 - 0.3877 * X3
    g4 = 0.66864 + 0.80172 * X - 0.75795 * X2 + 0.18895 * X3
    EghPSIM = Eext * Fbgw * F2PG * (g0 + g1 * S + g2 * S2 + g3 * S3 + g4 * S4) * F3RO

    # diffuse horizontal irradiance
    EdhPSIM = np.maximum(0, EghPSIM - EbnPSIM * np.cos(sza))

    # Quality control
    lower = 0
    EbnPSIM[EbnPSIM < lower] = 0
    EdhPSIM[EdhPSIM < lower] = 0
    EghPSIM[EghPSIM < lower] = 0

    return [EghPSIM, EbnPSIM, EdhPSIM]