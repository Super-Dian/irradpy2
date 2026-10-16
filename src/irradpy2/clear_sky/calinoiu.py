import numpy as np
import pandas as pd


def datetime_convert(datetime):
    """Convert a datetime sequence to (day-of-year, year) arrays.

    Parameters
    ----------
    datetime : array-like
        Datetime sequence parsed with ``pandas.to_datetime``; invalid entries
        become NaT. The original array shape is preserved.

    Returns
    -------
    Dayth : numpy.ndarray of float
        Day of year (1-366), same shape as ``datetime``.
    Year : numpy.ndarray of float
        Calendar year, same shape as ``datetime``.
    """
    datetime_array = np.asarray(datetime)
    original_shape = datetime_array.shape
    datetime_flat = datetime_array.reshape(-1)
    time = pd.to_datetime(datetime_flat, errors="coerce")
    Dayth = np.asarray(time.dayofyear, dtype=float).reshape(original_shape)
    Year = np.asarray(time.year, dtype=float).reshape(original_shape)

    return Dayth, Year


def calinoiu(sza: np.ndarray, datetime: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, NO2: np.ndarray, wv: np.ndarray):
    """
    Original Calinoiu clear-sky model.

    References
    ----------
    Calinoiu, D., Stefu, N., Boata, R., Blaga, R., Pop, N., Paulescu, E., ... & Paulescu, M. (2018).
    Parametric modeling: A simple and versatile route to solar irradiance.
    Energy Conversion and Management, 164, 175-187.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    NO2 : np.ndarray
        Total columnar amount of nitrogen dioxide [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghCalinoiu, EbnCalinoiu, EdhCalinoiu] : list of np.ndarray
        EghCalinoiu : Global horizontal irradiance [W m-2]
        EbnCalinoiu : Direct normal irradiance [W m-2]
        EdhCalinoiu : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    NO2 = np.asarray(NO2, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # beam normal irradiance
    # ozone
    mo3 = 1 / (np.cos(sza) + 268.45 * ((sza / np.pi * 180) ** 0.5) * (115.42 - sza / np.pi * 180) ** (-3.2922))
    xo3 = mo3 * ozone
    To3 = (1 + 8.5951 * xo3 + 0.2179 * xo3 ** 2) / (1 + 8.75308 * xo3 + 0.45 * xo3 ** 2 - 0.0004 * xo3 ** 3)

    # nitrogen dioxide
    mno2 = 1 / (np.cos(sza) + 602.3 * ((sza / np.pi * 180) ** 0.5) * (117.96 - sza / np.pi * 180) ** (-3.4536))
    xno2 = mno2 * NO2
    Tno2 = 0.742 + 0.258 * np.exp(-xno2 / 0.09)

    # water vapour
    mw = 1 / (np.cos(sza) + 0.031141 * ((sza / np.pi * 180) ** 0.1) * (92.471 - sza / np.pi * 180) ** (-1.3814))
    xw = mw * wv
    Tw = (1 + 1.4 * xw ** 0.5 + 0.053 * xw) * (1 - 0.013 * xw) / (1 + 1.626414 * xw ** 0.5 + 0.10816267 * xw)

    # trace gas absorption
    mg = 1 / (np.cos(sza) + 0.45665 * ((sza / np.pi * 180) ** 0.07) * (96.4836 - sza / np.pi * 180) ** (-1.6970))
    xg = mg
    Tg = (1 + 0.19558 * xg ** 0.5) / (1 + 0.215582 * xg ** 0.5 + 0.0005 * xg)

    # Rayleigh scattering
    mR = 1 / (np.cos(sza) + 0.45665 * ((sza / np.pi * 180) ** 0.07) * (96.4836 - sza / np.pi * 180) ** (-1.6970))
    xR = mR
    TR = (1 + 0.1564 * xR + 0.0001 * xR ** 2) / (1 + 0.26038 * xR + 0.00697 * xR ** 2)

    # aerosol extinction
    ma = 1 / (np.cos(sza) + 0.031141 * ((sza / np.pi * 180) ** 0.1) * (92.471 - sza / np.pi * 180) ** (-1.3814))
    xa = ma * ang_beta
    Ta = (1 - 0.046 * xa) / (1 + 1.73849 * xa + 0.79081 * xa ** 2)

    taub = To3 * Tno2 * Tw * Tg * TR * Ta

    # direct beam irradiance
    EbnCalinoiu = Eext * taub

    # diffuse irradiance
    # aerosol scattering
    g = 0.5
    f1g05 = 1.459 * np.log(1 - g) + 0.1595 * (np.log(1 - g)) ** 2 + 0.4129 * (np.log(1 - g)) ** 3
    f2g05 = 0.0783 * np.log(1 - g) - 0.3824 * (np.log(1 - g)) ** 2 + 0.5874 * (np.log(1 - g)) ** 3
    gamma_a05 = 0.92 * (1 - 0.5 * np.exp(f1g05 * np.cos(sza) + f2g05 * (np.cos(sza)) ** 2))
    g = 0.75
    f1g075 = 1.459 * np.log(1 - g) + 0.1595 * (np.log(1 - g)) ** 2 + 0.4129 * (np.log(1 - g)) ** 3
    f2g075 = 0.0783 * np.log(1 - g) - 0.3824 * (np.log(1 - g)) ** 2 + 0.5874 * (np.log(1 - g)) ** 3
    gamma_a075 = 0.92 * (1 - 0.5 * np.exp(f1g075 * np.cos(sza) + f2g075 * (np.cos(sza)) ** 2))
    gamma_a = 0.5 * (gamma_a05 + gamma_a075)

    omega_hat = 0.95
    Tas = (0.96865 - 1.57909 * np.log(omega_hat) - 0.07119 * xa + 2.78513 * (np.log(omega_hat)) ** 2 + 0.00157 * xa ** 2 + 0.07939 * xa * np.log(omega_hat)) / (1 - 1.22067 * np.log(omega_hat) + 1.83599 * xa + 3.20558 * (np.log(omega_hat)) ** 2 + 0.55453 * xa ** 2 - 1.73402 * xa * np.log(omega_hat))
    Taa = Ta / Tas * (0.14578 + 0.69109 / omega_hat)

    mo3[mo3 >= 17] = 16.999999999
    mo3[mo3 <= 0.001] = 0.001
    xo3[xo3 >= 0.45] = 0.449999999
    xo3[xo3 <= 0.15] = 0.150000001
    xo3_1 = -1 + 2 / 0.3 * (xo3 - 0.15)
    mo3_1 = -1 + 2 / 17 * (mo3 - 0.01)
    xo3_1[xo3_1 == 1] = 0.9999999
    mo3_1[mo3_1 == 1] = 0.9999999

    cij = np.zeros((10, 10), dtype=float)
    cij[0, :] = [0.907, -0.031, 1.026e-3, -1.106e-4, 1.703e-5, -4.738e-6, 4.409e-6, 1.145e-6, -1.49e-6, 1.748e-8]
    cij[1, :] = [-0.044, -0.014, 8.96e-4, -8.85e-5, 5.172e-6, -3.739e-6, -7.45e-7, -3.646e-6, 1.707e-6, 7.464e-8]
    cij[2, :] = [0.023, 7.475e-3, -3.653e-4, 3.38e-5, -1.041e-6, -4.724e-6, -8.628e-7, 2.768e-7, 2.305e-6, 0]
    cij[3, :] = [-0.012, -4.014e-3, 1.514e-4, -1.806e-5, 2.288e-6, -2.995e-6, -3.292e-6, -1.753e-6, 0, 0]
    cij[4, :] = [6.22e-3, 2.046e-3, -5.799e-5, 8.207e-6, -1.759e-6, -1.807e-6, 8.908e-7, 0, 0, 0]
    cij[5, :] = [3.158e-3, -9.618e-4, 2.104e-5, -4.049e-6, 3.605e-6, -1.828e-6, 0, 0, 0, 0]
    cij[6, :] = [1.576e-3, 4.2e-4, -1.083e-5, 7.229e-6, -3.751e-9, 0, 0, 0, 0, 0]
    cij[7, :] = [-8.093e-4, -1.768e-4, 6.057e-6, -1.74e-6, 0, 0, 0, 0, 0, 0]
    cij[8, :] = [4.428e-4, 6.802e-5, -4.685e-6, 0, 0, 0, 0, 0, 0, 0]
    cij[9, :] = [-2.212e-4, -3.167e-5, 0, 0, 0, 0, 0, 0, 0, 0]

    Td_o3 = np.zeros_like(sza, dtype=float)
    for i in range(10):
        for j in range(10 - i):
            Xjxo3 = np.cos(j * np.arccos(xo3_1))
            Ximo3 = np.cos(i * np.arccos(mo3_1))
            Td_o3 = Td_o3 + cij[i, j] * Xjxo3 * Ximo3

    tauda = gamma_a * Td_o3 * Tno2 * Tw * Tg * TR * Taa * (1 - Tas)

    # Rayleigh
    taudR = 0.5 * Td_o3 * Tno2 * Tw * Tg * Taa * (1 - TR)

    # diffuse horizontal irradiance
    EdhCalinoiu = Eext * np.cos(sza) * (taudR + tauda)

    # global irradiance
    EghCalinoiu = EbnCalinoiu * np.cos(sza) + EdhCalinoiu

    # Quality control
    lower = 0
    EbnCalinoiu[EbnCalinoiu < lower] = 0
    EdhCalinoiu[EdhCalinoiu < lower] = 0
    EghCalinoiu[EghCalinoiu < lower] = 0

    return [EghCalinoiu, EbnCalinoiu, EdhCalinoiu]