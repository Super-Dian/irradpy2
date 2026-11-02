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


def iqbal_b(sza: np.ndarray, datetime, pressure: np.ndarray, wv: np.ndarray, ozone: np.ndarray, ang_beta: np.ndarray, albedo: np.ndarray):
    """
    Iqbal-B 1983 clear-sky model.

    References
    ----------
    Iqbal, M. (1983).
    An Introduction to Solar Radiation.
    Chapter 7.5, Model B.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input. It is used to calculate Dayth and Year automatically.
    pressure : np.ndarray
        Local barometric pressure [mb]
    wv : np.ndarray
        Total precipitable water vapour [cm]
    ozone : np.ndarray
        Total ozone amount [atm-cm]
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    albedo : np.ndarray
        Ground albedo

    Returns
    -------
    [EghIQBALB, EbnIQBALB, EdhIQBALB] : list of np.ndarray
        EghIQBALB : Global horizontal irradiance [W m-2]
        EbnIQBALB : Direct normal irradiance [W m-2]
        EdhIQBALB : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    sza_deg = sza / np.pi * 180
    am = (np.cos(sza) + 0.15 * (93.885 - sza_deg) ** (-1.253)) ** (-1)
    ama = am * pressure / 1013.25

    # absorption ratio for water vapour
    U1 = am * wv * (pressure / 1013.25) ** 0.75
    aw = 0.110 * (U1 + 6.31e-4) ** 0.3 - 0.0121

    # absorption ratio for uniformly mixed gases
    ag = 0.00235 * (126 * ama + 0.0129) ** 0.26 - 7.5e-4 + 7.5e-3 * ama ** 0.875

    # absorption ratio for ozone
    U3 = ozone * am
    ao = 0.045 * (U3 + 8.34e-4) ** 0.38 - 3.1e-3

    # transmittance due to Rayleigh scattering
    Tr = 0.615958 + 0.375566 * np.exp(-0.221185 * ama)

    # transmittance due to scattering by aerosols
    gAB = -0.914 + 1.909267 * np.exp(-0.667023 * ang_beta)
    Tas = gAB ** ama

    # transmittance due to absorptance of aerosols
    w0 = 0.95
    Aa = (1 - w0) * gAB ** ama

    # direct normal irradiance
    EbnIQBALB = Eext * (1 - aw - ag - ao - Aa) * Tr * Tas

    # diffuse irradiance from the Rayleigh atmosphere
    Idr = Eext * np.cos(sza) * (1 - aw - ag - ao - Aa) * 0.5 * (1 - Tr)

    # diffuse irradiance from aerosol scattering
    Ida = Eext * np.cos(sza) * (1 - aw - ag - ao - Aa) * 0.75 * (1 - Tas)

    # diffuse irradiance from multiple reflections between ground and cloudless-sky atmosphere
    Q = EbnIQBALB * np.cos(sza) + Idr + Ida
    Tr1 = 0.615958 + 0.375566 * np.exp(-0.221185 * 1.66 * pressure / 1013.25)
    Tas1 = gAB ** (1.66 * pressure / 1013.25)
    U11 = wv * (pressure / 1013.25) ** 0.75 * (ama + 1.66 * pressure / 1013.25)
    aw1 = 0.110 * (U11 + 6.31e-4) ** 0.3 - 0.0121
    ag1 = 0.00235 * (126 * (ama + 1.66 * pressure / 1013.25) + 0.0129) ** 0.26 - 7.5e-4 + 7.5e-3 * (ama + 1.66 * pressure / 1013.25) ** 0.875
    U33 = ozone * (ama + 1.66 * pressure / 1013.25)
    ao1 = 0.045 * (U33 + 8.34e-4) ** 0.38 - 3.1e-3
    Aa1 = (1 - w0) * gAB ** (ama + 1.66 * pressure / 1013.25)
    Idm = albedo * Q * (1 - aw1 - ag1 - ao1 - Aa1) * (0.5 * (1 - Tr1) + 0.25 * (1 - Tas1))

    # diffuse horizontal irradiance
    EdhIQBALB = Idr + Ida + Idm

    # global horizontal irradiance
    EghIQBALB = EbnIQBALB * np.cos(sza) + EdhIQBALB

    # Quality control
    lower = 0
    EbnIQBALB[EbnIQBALB < lower] = lower
    EbnIQBALB[np.imag(EbnIQBALB) != 0] = np.nan
    EdhIQBALB[EdhIQBALB < lower] = lower
    EdhIQBALB[np.imag(EdhIQBALB) != 0] = np.nan
    EghIQBALB[EghIQBALB < lower] = lower
    EghIQBALB[np.imag(EghIQBALB) != 0] = np.nan

    return [EghIQBALB, EbnIQBALB, EdhIQBALB]