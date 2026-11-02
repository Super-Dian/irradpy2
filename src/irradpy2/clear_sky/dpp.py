import numpy as np

def dpp(sza: np.ndarray):
    """
    Original DPP clear-sky model.

    References
    ----------
    Daneshyar, M. (1978).
    Solar radiation statistics for Iran.
    Sol. Energy;(United States), 21(4).

    Badescu, V., Gueymard, C. A., Cheval, S., Oprea, C., Baciu, M.,
    Dumitrescu, A., ... & Rada, C. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models
    using routine hourly global irradiance measurements in Romania.
    Renewable Energy, 55, 85-103.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]

    Returns
    -------
    [Eghdpp, Ebndpp, Edhdpp] : list of np.ndarray
        Eghdpp : Global horizontal irradiance [W m-2]
        Ebndpp : Direct normal irradiance [W m-2]
        Edhdpp : Diffuse horizontal irradiance [W m-2]
    """

    # -------------------------
    # Make safe local arrays
    # -------------------------
    sza = np.asarray(sza, dtype=float)

    # direct normal irradiance
    Ebndpp = 950 * (1 - np.exp(-0.075 * (np.pi / 2 - sza) / np.pi * 180))

    # diffuse horizontal irradiance
    Edhdpp = 2.534 + 3.475 * (np.pi / 2 - sza) / np.pi * 180

    # global horizontal irradiance
    Eghdpp = Ebndpp * np.cos(sza) + Edhdpp

    # Quality control
    lower = 0
    Ebndpp[Ebndpp < lower] = 0
    Edhdpp[Edhdpp < lower] = 0
    Eghdpp[Eghdpp < lower] = 0

    return [Eghdpp, Ebndpp, Edhdpp]