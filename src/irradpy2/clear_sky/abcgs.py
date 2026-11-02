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


def abcgs(sza: np.ndarray):
    """
    ABCGS 1979 clear-sky model.

    References
    ----------
    Schulze, R. E. (1976).
    A physically based method of estimating solar radiation from suncards.

    Adnot, J., Bourges, B., Campana, D., & Gicquel, R. (1979).
    Utilisation des courbes de frequence cumulees pour le calcul des
    installations solaires. In R. Lestienne (Ed.), Analyse statistique
    des processus meteorologiques appliquee a l'energie solaire.
    CNRS, Paris.

    Badescu, V. (2013).
    Accuracy analysis for fifty-four clear-sky solar radiation models
    using routine hourly global irradiance measurements in Romania.

    Inputs
    ------
    sza : array-like
        Solar zenith angle [radians]

    Outputs
    -------
    [EghABCGS, EbnABCGS, EdhABCGS]
        EghABCGS : Global horizontal irradiance [W m-2]
        EbnABCGS : Direct normal irradiance [W m-2]
        EdhABCGS : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    lower = 0

    # direct normal irradiance
    EbnABCGS = 951.39 * np.cos(sza) ** 1.15 / np.cos(sza)
    EbnABCGS[EbnABCGS < lower] = lower

    # diffuse horizontal irradiance
    EdhABCGS = 94.23 * np.cos(sza) ** 0.5
    EdhABCGS[EdhABCGS < lower] = lower

    # global horizontal irradiance
    EghABCGS = EbnABCGS * np.cos(sza) + EdhABCGS
    EghABCGS[EghABCGS < lower] = lower

    return [EghABCGS, EbnABCGS, EdhABCGS]