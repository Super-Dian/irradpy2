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


def rest2v5(sza: np.ndarray, datetime: np.ndarray, press: np.ndarray, albedo: np.ndarray, ang_alpha: np.ndarray, ang_beta: np.ndarray, ozone: np.ndarray, NO2: np.ndarray, wv: np.ndarray):
    """
    Original REST2 version 5 clear-sky model.

    References
    ----------
    Gueymard, C. A. (2008). REST2: High-performance solar radiation model for cloudless-sky irradiance, illuminance,
    and photosynthetically active radiation–Validation with a benchmark dataset.
    Solar Energy, 82(3), 272-285.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    Dayth : np.ndarray
        Day number ranging from 1 to 365
    Year : np.ndarray
        Year
    press : np.ndarray
        Local barometric pressure [mb]
    albedo : np.ndarray
        Surface/ground albedo
    ang_alpha : np.ndarray
        Angstrom exponent
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
    [Eghrest2, Ebnrest2, Edhrest2] : list of np.ndarray
        Eghrest2 : Global horizontal irradiance [W m-2]
        Ebnrest2 : Direct normal irradiance [W m-2]
        Edhrest2 : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    Dayth = np.asarray(Dayth, dtype=float)
    Year = np.asarray(Year, dtype=float)
    press = np.asarray(press, dtype=float)
    albedo = np.asarray(albedo, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    NO2 = np.asarray(NO2, dtype=float)
    wv = np.asarray(wv, dtype=float)

    # extraterrestrial irradiance
    Esc = 1366.1
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    Eext = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # air mass for aerosols extinction
    ama = (np.cos(sza) + 0.16851 * (sza * 180 / np.pi) ** 0.18198 / (95.318 - sza * 180 / np.pi) ** 1.9542) ** -1

    # air mass for water vapor absorption
    amw = (np.cos(sza) + 0.10648 * (sza * 180 / np.pi) ** 0.11423 / (93.781 - sza * 180 / np.pi) ** 1.9203) ** -1

    # air mass for nitrogen dioxide absorption
    amn = (np.cos(sza) + 1.1212 * (sza * 180 / np.pi) ** 1.6132 / (111.55 - sza * 180 / np.pi) ** 3.2629) ** -1

    # air mass for ozone absorption
    amo = (np.cos(sza) + 1.0651 * (sza * 180 / np.pi) ** 0.6379 / (101.8 - sza * 180 / np.pi) ** 2.2694) ** -1

    # air mass for Rayleigh scattering and uniformly mixed gases absorption
    amR = (np.cos(sza) + 0.48353 * ((sza * 180 / np.pi) ** 0.095846) / (96.741 - sza * 180 / np.pi) ** 1.754) ** -1
    amRe = (press / 1013.25) * (np.cos(sza) + 0.48353 * ((sza * 180 / np.pi) ** 0.095846) / (96.741 - sza * 180 / np.pi) ** 1.754) ** -1

    # band 1
    # transmittance for Rayleigh scattering
    TR1 = (1 + 1.8169 * amRe - 0.033454 * amRe ** 2) / (1 + 2.063 * amRe + 0.31978 * amRe ** 2)

    # transmittance for uniformly mixed gases absorption
    Tg1 = (1 + 0.95885 * amRe + 0.012871 * amRe ** 2) / (1 + 0.96321 * amRe + 0.015455 * amRe ** 2)

    # transmittance for ozone absorption
    uo = ozone
    f1 = uo * (10.979 - 8.5421 * uo) / (1 + 2.0115 * uo + 40.189 * uo ** 2)
    f2 = uo * (-0.027589 - 0.005138 * uo) / (1 - 2.4857 * uo + 13.942 * uo ** 2)
    f3 = uo * (10.995 - 5.5001 * uo) / (1 + 1.6784 * uo + 42.406 * uo ** 2)
    To1 = (1 + f1 * amo + f2 * amo ** 2) / (1 + f3 * amo)

    # transmittance for nitrogen dioxide absorption
    un = NO2
    g1 = (0.17499 + 41.654 * un - 2146.4 * un ** 2) / (1 + 22295 * un ** 2)
    g2 = un * (-1.2134 + 59.324 * un) / (1 + 8847.8 * un ** 2)
    g3 = (0.17499 + 61.658 * un + 9196.4 * un ** 2) / (1 + 74109 * un ** 2)
    Tn1 = np.minimum(1, (1 + g1 * amw + g2 * amw ** 2) / (1 + g3 * amw))
    Tn1166 = np.minimum(1, (1 + g1 * 1.66 + g2 * 1.66 ** 2) / (1 + g3 * 1.66))

    # transmittance for water vapor absorption
    h1 = wv * (0.065445 + 0.00029901 * wv) / (1 + 1.2728 * wv)
    h2 = wv * (0.065687 + 0.0013218 * wv) / (1 + 1.2008 * wv)
    Tw1 = (1 + h1 * amw) / (1 + h2 * amw)
    Tw1166 = (1 + h1 * 1.66) / (1 + h2 * 1.66)

    # coefficients of angstrom alpha
    AB1 = ang_beta
    alph1 = ang_alpha
    d0 = 0.57664 - 0.024743 * alph1
    d1 = (0.093942 - 0.2269 * alph1 + 0.12848 * alph1 ** 2) / (1 + 0.6418 * alph1)
    d2 = (-0.093819 + 0.36668 * alph1 - 0.12775 * alph1 ** 2) / (1 - 0.11651 * alph1)
    d3 = alph1 * (0.15232 - 0.087214 * alph1 + 0.012664 * alph1 ** 2) / (1 - 0.90454 * alph1 + 0.26167 * alph1 ** 2)
    ua1 = np.log(1 + ama * AB1)
    lam1 = (d0 + d1 * ua1 + d2 * ua1 ** 2) / (1 + d3 * ua1 ** 2)

    # aerosol transmittance
    ta1 = AB1 * lam1 ** (-alph1)
    TA1 = np.exp(-ama * ta1)

    # aerosol scattering transmittance
    TAS1 = np.exp(-ama * 0.92 * ta1)

    # forward scattering fractions for Rayleigh extinction
    BR1 = 0.5 * (0.89013 - 0.0049558 * amR + 0.000045721 * amR ** 2)

    # the aerosol forward scatterance factor
    Ba = 1 - np.exp(-0.6931 - 1.8326 * np.cos(sza))

    # aerosol scattering correction factor
    g0 = (3.715 + 0.368 * ama + 0.036294 * ama ** 2) / (1 + 0.0009391 * ama ** 2)
    g1 = (-0.164 - 0.72567 * ama + 0.20701 * ama ** 2) / (1 + 0.0019012 * ama ** 2)
    g2 = (-0.052288 + 0.31902 * ama + 0.17871 * ama ** 2) / (1 + 0.0069592 * ama ** 2)
    F1 = (g0 + g1 * ta1) / (1 + g2 * ta1)

    # sky albedo
    rs1 = (0.13363 + 0.00077358 * alph1 + AB1 * (0.37567 + 0.22946 * alph1) / (1 - 0.10832 * alph1)) / (1 + AB1 * (0.84057 + 0.68683 * alph1) / (1 - 0.08158 * alph1))
    rg = albedo

    # band 2
    # transmittance for Rayleigh scattering
    TR2 = (1 - 0.010394 * amRe) / (1 - 0.00011042 * amRe ** 2)

    # transmittance for uniformly mixed gases absorption
    Tg2 = (1 + 0.27284 * amRe - 0.00063699 * amRe ** 2) / (1 + 0.30306 * amRe)

    # transmittance for ozone absorption
    To2 = 1

    # transmittance for nitrogen dioxide absorption
    Tn2 = 1
    Tn2166 = 1

    # transmittance for water vapor absorption
    c1 = wv * (19.566 - 1.6506 * wv + 1.0672 * wv ** 2) / (1 + 5.4248 * wv + 1.6005 * wv ** 2)
    c2 = wv * (0.50158 - 0.14732 * wv + 0.047584 * wv ** 2) / (1 + 1.1811 * wv + 1.0699 * wv ** 2)
    c3 = wv * (21.286 - 0.39232 * wv + 1.2692 * wv ** 2) / (1 + 4.8318 * wv + 1.412 * wv ** 2)
    c4 = wv * (0.70992 - 0.23155 * wv + 0.096514 * wv ** 2) / (1 + 0.44907 * wv + 0.75425 * wv ** 2)
    Tw2 = (1 + c1 * amw + c2 * amw ** 2) / (1 + c3 * amw + c4 * amw ** 2)
    Tw2166 = (1 + c1 * 1.66 + c2 * 1.66 ** 2) / (1 + c3 * 1.66 + c4 * 1.66 ** 2)

    # coefficients of angstrom alpha
    AB2 = ang_beta
    alph2 = ang_alpha
    e0 = (1.183 - 0.022989 * alph2 + 0.020829 * alph2 ** 2) / (1 + 0.11133 * alph2)
    e1 = (-0.50003 - 0.18329 * alph2 + 0.23835 * alph2 ** 2) / (1 + 1.6756 * alph2)
    e2 = (-0.50001 + 1.1414 * alph2 + 0.0083589 * alph2 ** 2) / (1 + 11.168 * alph2)
    e3 = (-0.70003 - 0.73587 * alph2 + 0.51509 * alph2 ** 2) / (1 + 4.7665 * alph2)
    ua2 = np.log(1 + ama * AB2)
    lam2 = (e0 + e1 * ua2 + e2 * ua2 ** 2) / (1 + e3 * ua2)

    # aerosol transmittance
    ta2 = AB2 * lam2 ** (-alph2)
    TA2 = np.exp(-ama * ta2)
    TAS2 = np.exp(-ama * 0.84 * ta2)

    # forward scattering fractions for Rayleigh extinction
    BR2 = 0.5

    # the aerosol forward scatterance factor
    Ba = 1 - np.exp(-0.6931 - 1.8326 * np.cos(sza))

    # aerosol scattering correction
    h0 = (3.4352 + 0.65267 * ama + 0.00034328 * ama ** 2) / (1 + 0.034388 * ama ** 1.5)
    h1 = (1.231 - 1.63853 * ama + 0.20667 * ama ** 2) / (1 + 0.1451 * ama ** 1.5)
    h2 = (0.8889 - 0.55063 * ama + 0.50152 * ama ** 2) / (1 + 0.14865 * ama ** 1.5)
    F2 = (h0 + h1 * ta2) / (1 + h2 * ta2)

    # sky albedo
    rs2 = (0.010191 + 0.00085547 * alph2 + AB2 * (0.14618 + 0.062758 * alph2) / (1 - 0.19402 * alph2)) / (1 + AB2 * (0.58101 + 0.17426 * alph2) / (1 - 0.17586 * alph2))
    rg = albedo

    # irradiance band 1
    E0n1 = Eext * 0.46512
    Ebn1 = E0n1 * TR1 * Tg1 * To1 * Tn1 * Tw1 * TA1
    Edp1 = E0n1 * np.cos(sza) * To1 * Tg1 * Tn1166 * Tw1166 * (BR1 * (1 - TR1) * TA1 ** 0.25 + Ba * F1 * TR1 * (1 - TAS1 ** 0.25))
    Edd1 = rg * rs1 * (Ebn1 * np.cos(sza) + Edp1) / (1 - rg * rs1)

    # irradiance band 2
    E0n2 = Eext * 0.51951
    Ebn2 = E0n2 * TR2 * Tg2 * To2 * Tn2 * Tw2 * TA2
    Edp2 = E0n2 * np.cos(sza) * To2 * Tg2 * Tn2166 * Tw2166 * (BR2 * (1 - TR2) * TA2 ** 0.25 + Ba * F2 * TR2 * (1 - TAS2 ** 0.25))
    Edd2 = rg * rs2 * (Ebn2 * np.cos(sza) + Edp2) / (1 - rg * rs2)

    # totals band 1 + band 2
    Ebnrest2 = Ebn1 + Ebn2
    Edhrest2 = Edp1 + Edd1 + Edp2 + Edd2
    Eghrest2 = Ebnrest2 * np.cos(sza) + Edhrest2

    # Quality control
    lower = 0
    Ebnrest2[Ebnrest2 < lower] = 0
    Edhrest2[Edhrest2 < lower] = 0
    Eghrest2[Eghrest2 < lower] = 0

    return [Eghrest2, Ebnrest2, Edhrest2]