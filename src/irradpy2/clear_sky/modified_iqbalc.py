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


def modified_iqbalc(sza: np.ndarray, datetime, altitude: np.ndarray, albedo: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, wv: np.ndarray):
    """
    Original modified Iqbal-C clear-sky model.

    References
    ----------
    Bird, R. E., & Hulstrom, R. L. (1981). Simplified clear sky model for direct and diffuse insolation on horizontal surfaces
    (No. SERI/TR-642-761). Solar Energy Research Inst., Golden, CO (USA).
    Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
    Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
    Solar Energy, 86(8), 2145-2169.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : np.ndarray
        Datetime data; day of year and year are derived from it internally
    altitude : np.ndarray
        Location altitude [m]
    albedo : np.ndarray
        Surface/ground albedo
    ang_alpha : np.ndarray
        Angstrom exponent
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ozone : np.ndarray
        Total columnar amount of ozone [atm.cm]
    wv : np.ndarray
        Total columnar amount of water vapour [atm.cm]

    Returns
    -------
    [EghMiqbalc, EbnMiqbalc, EdhMiqbalc] : list of np.ndarray
        EghMiqbalc : Global horizontal irradiance [W m-2]
        EbnMiqbalc : Direct normal irradiance [W m-2]
        EdhMiqbalc : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    altitude = np.asarray(altitude, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
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
    ama = am * np.exp(-0.0001184 * altitude)

    # the transmittance by Rayleigh scattering
    TR = np.exp(-0.0903 * ama ** 0.84 * (1 + ama - ama ** 1.01))

    # the transmittance by ozone
    U3 = ozone * am
    alpho = 0.1611 * U3 * (1 + 139.48 * U3) ** -0.3035 - 0.002715 * U3 * (1.0 + 0.044 * U3 + 0.0003 * U3 ** 2) ** -1
    TO = 1 - alpho

    # the transmittance by uniformly mixed gases
    TG = np.exp(-0.0127 * ama ** 0.26)

    # the transmittance by water vapor
    PW = wv * am
    alphw = 2.4959 * PW * ((1 + 79.034 * PW) ** 0.6828 + 6.385 * PW) ** -1
    TW = 1 - alphw

    # the transmittance by aerosol
    TA = (0.12445 * ang_alpha - 0.0162) + (1.003 - 0.125 * ang_alpha) * np.exp(-ama * ang_beta * (1.089 * ang_alpha + 0.5123))

    # direct beam irradiance
    EbnMiqbalc = 0.9751 * Eext * TR * TO * TG * TW * TA

    # the transmittance of direct radiation due to aerosol absorptance
    w0 = 0.9
    TAA = 1 - (1 - w0) * (1 - ama + ama ** 1.06) * (1 - TA)

    # the Rayleigh-scattered diffuse irradiance after the first pass through the atmosphere
    Edr = 0.79 * Eext * np.cos(sza) * TO * TG * TW * TAA * 0.5 * (1 - TR) / (1 - ama + ama ** 1.02)

    # the aerosol scattered diffuse irradiance after the first pass through the atmosphere
    TAS = TA / TAA
    Fc = 0.84
    Eda = 0.79 * Eext * np.cos(sza) * TO * TG * TW * TAA * Fc * (1 - TAS) / (1 - ama + ama ** 1.02)

    # global horizontal irradiance
    rg = albedo
    ra = 0.0685 + (1 - Fc) * (1 - TAS)
    EghMiqbalc = (EbnMiqbalc * np.cos(sza) + Edr + Eda) / (1 - rg * ra)

    # diffuse horizontal irradiance
    EdhMiqbalc = EghMiqbalc - EbnMiqbalc * np.cos(sza)

    # Quality control
    lower = 0
    EbnMiqbalc[EbnMiqbalc < lower] = 0
    EdhMiqbalc[EdhMiqbalc < lower] = 0
    EghMiqbalc[EghMiqbalc < lower] = 0

    return [EghMiqbalc, EbnMiqbalc, EdhMiqbalc]