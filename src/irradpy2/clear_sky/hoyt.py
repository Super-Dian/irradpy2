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


def hoyt(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, albedo: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original Hoyt clear-sky model.

    References
    ----------
    Iqbal, M. (2012). An introduction to solar radiation.
    Elsevier.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data. It is used to calculate Dayth and Year automatically.
    press : np.ndarray
        Local barometric pressure [mb]
    albedo : np.ndarray
        Surface/ground albedo
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [Eghhoyt, Ebnhoyt, Edhhoyt] : list of np.ndarray
        Eghhoyt : Global horizontal irradiance [W m-2]
        Ebnhoyt : Direct normal irradiance [W m-2]
        Edhhoyt : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1367
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass
    am = (np.cos(sza) + 0.15 * (3.885 + (np.pi / 2 - sza) / np.pi * 180) ** -1.253) ** -1
    ama = am * press / 1013.25

    # absorption ratio for water vapor
    U1 = am * wv * (press / 1013.25) ** 0.75
    aw = 0.110 * (U1 + 6.31e-4) ** 0.3 - 0.0121

    # absorption ratio for uniformly mixed gases
    ag = 0.00235 * (126 * ama + 0.0129) ** 0.26 - 7.5e-4 + 7.5e-3 * ama ** 0.875

    # absorption ratio for ozone
    U3 = ozone * am
    ao = 0.045 * (U3 + 8.34e-4) ** 0.38 - 3.1e-3

    # the transmittance due to Rayleigh scattering
    Tr = 0.615958 + 0.375566 * np.exp(-0.221185 * ama)

    # the transmittance due to scattering by aerosols
    gAB = -0.914 + 1.909267 * np.exp(-0.667023 * ang_beta)
    Tas = gAB ** ama

    # the transmittance due to absorptance of aerosols
    w0 = 0.95
    Aa = (1 - w0) * gAB ** ama

    # direct beam irradiance
    Ebnhoyt = Eext * (1 - aw - ag - ao - Aa) * Tr * Tas

    # the diffuse irradiance from the Rayleigh atmosphere
    Idr = Eext * np.cos(sza) * (1 - aw - ag - ao - Aa) * 0.5 * (1 - Tr)

    # the diffuse irradiance from aerosol scattering
    Ida = Eext * np.cos(sza) * (1 - aw - ag - ao - Aa) * 0.75 * (1 - Tas)

    # the diffuse irradiance from multiple reflections between ground and cloudless-sky atmosphere
    Q = Ebnhoyt * np.cos(sza) + Idr + Ida
    Tr1 = 0.615958 + 0.375566 * np.exp(-0.221185 * 1.66 * press / 1013.25)
    Tas1 = gAB ** (1.66 * press / 1013.25)
    U11 = wv * (press / 1013.25) ** 0.75 * (ama + 1.66 * press / 1013.25)
    aw1 = 0.110 * (U11 + 6.31e-4) ** 0.3 - 0.0121
    ag1 = 0.00235 * (126 * (ama + 1.66 * press / 1013.25) + 0.0129) ** 0.26 - 7.5e-4 + 7.5e-3 * (ama + 1.66 * press / 1013.25) ** 0.875
    U33 = ozone * (ama + 1.66 * press / 1013.25)
    ao1 = 0.045 * (U33 + 8.34e-4) ** 0.38 - 3.1e-3
    Aa1 = (1 - w0) * gAB ** (ama + 1.66 * press / 1013.25)
    Idm = albedo * Q * (1 - aw1 - ag1 - ao1 - Aa1) * (0.5 * (1 - Tr1) + 0.25 * (1 - Tas1))

    # diffuse horizontal irradiance
    Edhhoyt = Idr + Ida + Idm

    # global horizontal irradiance
    Eghhoyt = Ebnhoyt * np.cos(sza) + Edhhoyt

    # Quality control
    lower = 0
    Ebnhoyt[Ebnhoyt < lower] = 0
    Edhhoyt[Edhhoyt < lower] = 0
    Eghhoyt[Eghhoyt < lower] = 0

    return [Eghhoyt, Ebnhoyt, Edhhoyt]