import numpy as np

def schulze(sza: np.ndarray):
    """
    Original Schulze clear-sky model.

    References
    ----------
    Schulze, R. E. (1976).
    A physically based method of estimating solar radiation from suncards.
    Agricultural Meteorology, 16(1), 85-101.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]

    Returns
    -------
    [Ebnschulze, Edhschulze, Eghschulze] : list of np.ndarray
        Ebnschulze : Direct normal irradiance [W m-2]
        Edhschulze : Diffuse horizontal irradiance [W m-2]
        Eghschulze : Global horizontal irradiance [W m-2]
    """

    # -------------------------
    # Make safe local arrays
    # -------------------------
    sza = np.asarray(sza, dtype=float)

    # direct normal irradiance
    Ebnschulze = 1127 * (0.888 ** (1 / np.cos(sza)))

    # diffuse horizontal irradiance
    Edhschulze = 94.23 * (np.cos(sza) ** 0.5)

    # global horizontal irradiance
    Eghschulze = Ebnschulze * np.cos(sza) + Edhschulze

    # Quality control
    lower = 0
    Ebnschulze[Ebnschulze < lower] = 0
    Edhschulze[Edhschulze < lower] = 0
    Eghschulze[Eghschulze < lower] = 0

    return [Ebnschulze, Edhschulze, Eghschulze]