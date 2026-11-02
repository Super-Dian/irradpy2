import numpy as np


def hlj(sza: np.ndarray, Dayth: np.ndarray, Year: np.ndarray, altitude: np.ndarray):
    """
    Original HLJ clear-sky model.

    References
    ----------
    Hottel, H. C. (1976). A simple model for estimating the transmittance of direct solar radiation through clear atmospheres.
    Solar Energy, 18(2), 129-134.

    Gueymard, C. A. (2012). Clear-sky irradiance predictions for solar resource mapping and large-scale applications:
    Improved validation methodology and detailed performance analysis of 18 broadband radiative models.
    Solar Energy, 86(8), 2145-2169.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    Dayth : np.ndarray
        Day number ranging from 1 to 365
    Year : np.ndarray
        Year
    altitude : np.ndarray
        Location altitude [m]

    Returns
    -------
    [Eghhlj, Ebnhlj, Edhhlj] : list of np.ndarray
        Eghhlj : Global horizontal irradiance [W m-2]
        Ebnhlj : Direct normal irradiance [W m-2]
        Edhhlj : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    altitude = np.asarray(altitude, dtype=float)

    # extraterrestrial irradiance
    Esc = 1353
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # solar transmittance a0 + a1 * exp(-a2 / cos(sza))
    a0 = 0.4237 - 0.00821 * (6 - altitude / 1000) ** 2
    a1 = 0.5055 + 0.00595 * (6.5 - altitude / 1000) ** 2
    a2 = np.where(altitude / 1000 <= 2.5, 0.2711 + 0.01858 * (2.5 - altitude / 1000) ** 2, -0.02173 + 3.3693e-4 * altitude)

    # direct beam irradiance
    Ebnhlj = Eext * (a0 + a1 * np.exp(-a2 / np.cos(sza)))

    # diffuse horizontal irradiance
    Edhhlj = (0.2710 * Eext - 0.2939 * Ebnhlj) * np.cos(sza)

    # global horizontal irradiance
    Eghhlj = Ebnhlj * np.cos(sza) + Edhhlj

    # Quality control
    lower = 0
    Ebnhlj[Ebnhlj < lower] = 0
    Edhhlj[Edhhlj < lower] = 0
    Eghhlj[Eghhlj < lower] = 0

    return [Eghhlj, Ebnhlj, Edhhlj]