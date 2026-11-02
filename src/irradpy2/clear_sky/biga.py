import numpy as np

def biga(sza: np.ndarray):
    """
    Original Biga clear-sky model.

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
    [EghBiga, EbnBiga, EdhBiga] : list of np.ndarray
        EghBiga : Global horizontal irradiance [W m-2]
        EbnBiga : Direct normal irradiance [W m-2]
        EdhBiga : Diffuse horizontal irradiance [W m-2]
    """

    # -------------------------
    # Make safe local arrays
    # -------------------------
    sza = np.asarray(sza, dtype=float)

    # direct normal irradiance
    EbnBiga = 926 * (np.cos(sza) ** 0.29)

    # diffuse horizontal irradiance
    EdhBiga = 131 * (np.cos(sza) ** 0.6)

    # global horizontal irradiance
    EghBiga = 1057 * (np.cos(sza) ** 1.17)

    # Quality control
    lower = 0
    EbnBiga[EbnBiga < lower] = 0
    EdhBiga[EdhBiga < lower] = 0
    EghBiga[EghBiga < lower] = 0

    return [EghBiga, EbnBiga, EdhBiga]