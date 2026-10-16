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


def heliosat2(sza: np.ndarray, datetime: np.ndarray, altitude: np.ndarray, TL2_series: list[np.ndarray]):
    """
    Original Heliosat-2 clear-sky model.

    References
    ----------
    Lefevre, M., Albuisson, M., & Wald, L. (2002).
    Joint report on interpolation scheme 'meteosat' and database 'climatology' i (meteosat).
    SoDa Deliverable D3-8 and D5-1-4. Internal document.

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
    [EghHeliosat2, EbnHeliosat2, EdhHeliosat2] : list
        EghHeliosat2 : list of global horizontal irradiance series
        EbnHeliosat2 : list of direct normal irradiance series
        EdhHeliosat2 : list of diffuse horizontal irradiance series
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
    am = 1 / (np.sin(alphatrue) + 0.50572 * ((alphatrue / np.pi * 180 + 6.07995) ** -1.6364))

    # correction of the integral Rayleigh optical thickness due to elevation of the site
    corr_rot1 = 1
    corr_rot075 = 1.248174 - 0.011997 * am + 0.00037 * am ** 2
    corr_rot05 = 1.68219 - 0.03059 * am + 0.00089 * am ** 2

    # piecewise linear interpolation
    corr_rot = np.array(corr_rot05, copy=True)
    xzuobiao = np.array([0.5, 0.75, 1.0], dtype=float)
    pchup0 = np.exp(-altitude / 8434.5)
    for i in range(corr_rot.size):
        if np.isnan(corr_rot05.flat[i]) and np.isnan(corr_rot075.flat[i]):
            continue
        corr_rot.flat[i] = np.interp(pchup0.flat[i], xzuobiao, np.array([corr_rot05.flat[i], corr_rot075.flat[i], corr_rot1], dtype=float))

    # Rayleigh Optical Thickness
    rot = (corr_rot ** -1) * (6.625928 + 1.92969 * am - 0.170073 * am ** 2 + 0.011517 * am ** 3 - 0.000285 * am ** 4) ** -1
    rot[am > 20] = (10.4 + 0.718 * am[am > 20] * np.exp(-altitude[am > 20] / 8434.5)) ** -1

    # piecewise coefficients
    L00 = np.array(alpha, copy=True)
    L00[alpha > np.pi / 6] = -1.7349e-2
    L00[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = -8.2193e-3
    L00[alpha <= np.pi / 12] = -1.1656e-3

    L01 = np.array(alpha, copy=True)
    L01[alpha > np.pi / 6] = -5.8985e-3
    L01[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = 4.5643e-4
    L01[alpha <= np.pi / 12] = 1.8408e-4

    L02 = np.array(alpha, copy=True)
    L02[alpha > np.pi / 6] = 6.8868e-4
    L02[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = 6.7916e-5
    L02[alpha <= np.pi / 12] = -4.8754e-7

    L10 = np.array(alpha, copy=True)
    L10[alpha > np.pi / 6] = 1.0258
    L10[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = 8.9233e-1
    L10[alpha <= np.pi / 12] = 7.4095e-1

    L11 = np.array(alpha, copy=True)
    L11[alpha > np.pi / 6] = -1.2196e-1
    L11[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = -1.9991e-1
    L11[alpha <= np.pi / 12] = -2.2427e-1

    L12 = np.array(alpha, copy=True)
    L12[alpha > np.pi / 6] = 1.9299e-3
    L12[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = 9.9741e-3
    L12[alpha <= np.pi / 12] = 1.5314e-2

    L20 = np.array(alpha, copy=True)
    L20[alpha > np.pi / 6] = -7.2178e-3
    L20[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = 2.5428e-1
    L20[alpha <= np.pi / 12] = 3.4959e-1

    L21 = np.array(alpha, copy=True)
    L21[alpha > np.pi / 6] = 1.3086e-1
    L21[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = 2.6140e-1
    L21[alpha <= np.pi / 12] = 7.2313e-1

    L22 = np.array(alpha, copy=True)
    L22[alpha > np.pi / 6] = -2.8405e-3
    L22[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = -1.7020e-2
    L22[alpha <= np.pi / 12] = -1.2305e-1

    L23 = np.array(alpha, copy=True)
    L23[alpha > np.pi / 6] = 0
    L23[(alpha > np.pi / 12) & (alpha <= np.pi / 6)] = 0
    L23[alpha <= np.pi / 12] = 5.9194e-3

    EghHeliosat2 = []
    EbnHeliosat2 = []
    EdhHeliosat2 = []
    lower = 0

    for TL2 in TL2_series:
        Trb = np.exp(-0.8662 * TL2 * np.exp(altitude / 8434.5) * rot)
        c0 = L00 + L01 * TL2 * pchup0 + L02 * (TL2 * pchup0) ** 2
        c1 = L10 + L11 * TL2 * pchup0 + L12 * (TL2 * pchup0) ** 2
        c2 = L20 + L21 * TL2 * pchup0 + L22 * (TL2 * pchup0) ** 2 + L23 * (TL2 * pchup0) ** 3
        Fb = c0 + c1 * np.sin(alpha) + c2 * (np.sin(alpha) ** 2)
        Ebn_i = Eext * np.maximum(Trb * Fb, 0) / np.sin(alpha)
        Ebn_i[Ebn_i < lower] = 0

        TRD = -1.5843e-2 + 3.0543e-2 * TL2 + 3.797e-4 * TL2 ** 2
        a01 = 2.6463e-1 - 6.1581e-2 * TL2 + 3.1408e-3 * TL2 ** 2
        a11 = 2.0402 + 1.8945e-2 * TL2 - 1.1161e-2 * TL2 ** 2
        a21 = -1.3025 + 3.9231e-2 * TL2 + 8.5079e-3 * TL2 ** 2
        xiabiao = a01 * TRD < 2e-3
        a01[xiabiao] = (2e-3) / TRD[xiabiao]
        FD = a01 + a11 * np.sin(alpha) + a21 * (np.sin(alpha) ** 2)
        Edh_i = Eext * np.maximum(0, TRD * FD)
        Edh_i[Edh_i < lower] = 0

        Egh_i = Ebn_i * np.cos(sza) + Edh_i
        Egh_i[Egh_i < lower] = 0

        EghHeliosat2.append(Egh_i)
        EbnHeliosat2.append(Ebn_i)
        EdhHeliosat2.append(Edh_i)

    return [EghHeliosat2, EbnHeliosat2, EdhHeliosat2]