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


def cpcr2(sza: np.ndarray, datetime, pressure: np.ndarray, wv: np.ndarray, ozone: np.ndarray, ang_beta: np.ndarray, ang_alpha: np.ndarray, albedo: np.ndarray):
    """
    CPCR2 1989 clear-sky model.

    References
    ----------
    Gueymard, C. (1989).
    A two-band model for the calculation of clear-sky solar irradiance,
    illuminance, and photosynthetically active radiation at the Earth's
    surface.

    Gueymard, C. (1993).
    CPCR2 model with revised optical masses.
    Solar Energy.

    Chris' Fortran code.

    Parameters
    ----------
    sza : np.ndarray
        Solar zenith angle [radians]
    datetime : array-like
        Datetime input
    pressure : np.ndarray
        Local barometric pressure [mb]
    wv : np.ndarray
        Total precipitable water vapour [cm]
    ozone : np.ndarray
        Total ozone amount [atm-cm]
    ang_beta : np.ndarray
        Angstrom turbidity coefficient
    ang_alpha : np.ndarray
        Angstrom exponent
    albedo : np.ndarray
        Ground albedo

    Returns
    -------
    [EghCPCR2, EbnCPCR2, EdhCPCR2] : list of np.ndarray
        EghCPCR2 : Global horizontal irradiance [W m-2]
        EbnCPCR2 : Direct normal irradiance [W m-2]
        EdhCPCR2 : Diffuse horizontal irradiance [W m-2]
    """

    sza = np.asarray(sza, dtype=float)
    Dayth, Year = datetime_convert(datetime)
    pressure = np.asarray(pressure, dtype=float)
    wv = np.asarray(wv, dtype=float)
    ozone = np.asarray(ozone, dtype=float)
    ang_beta = np.asarray(ang_beta, dtype=float)
    ang_alpha = np.asarray(ang_alpha, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    # extraterrestrial irradiance
    Esc = 1369
    totaldayofyear = 366 - np.ceil(Year / 4 - np.trunc(Year / 4))
    B = (Dayth - 1) * 2 * np.pi / totaldayofyear
    En0 = Esc * (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))

    # pre-set
    F1 = 0.4604
    F2 = 0.5057
    F12 = 0.9661
    p0 = 1013.25
    pp0 = pressure / p0
    AMd = 1.66
    AMdp = AMd * pp0
    ALFA1 = ang_alpha
    ALFA12 = ALFA1 * ALFA1
    ALFA2 = ang_alpha
    ALFA22 = ALFA2 * ALFA2
    BETA = ang_beta
    W = wv
    Uo = ozone
    PIAR1 = 0.965
    PIAR2 = 0.913
    ROG = albedo

    # optical masses
    sza_deg = sza / np.pi * 180
    H = 90 - sza_deg
    FMR = (H + 4.37515) ** 1.21563
    FMW = (H + 2.24849) ** 1.2529
    AM = 1 / (np.cos(sza) + 0.00176759 * sza_deg / FMR)
    AM2 = AM * AM
    AMp = AM * pp0
    AMW = 1 / (np.cos(sza) + 0.000429452 * sza_deg / FMW)
    AMZ = 13.5 / (181.25 * np.cos(sza) * np.cos(sza) + 1) ** 0.5

    Al01 = 0.48530305 * np.ones_like(sza)
    Al01[np.abs(ALFA1 - 1.3) > 1e-3] = 0.510941 - 0.028607 * ALFA1[np.abs(ALFA1 - 1.3) > 1e-3] + 0.006835 * ALFA12[np.abs(ALFA1 - 1.3) > 1e-3]
    Al11 = 0.05603278 * np.ones_like(sza)
    Al11[np.abs(ALFA1 - 1.3) > 1e-3] = -0.026895 + 0.054857 * ALFA1[np.abs(ALFA1 - 1.3) > 1e-3] + 0.006872 * ALFA12[np.abs(ALFA1 - 1.3) > 1e-3]
    Al21 = 0.00104599 * np.ones_like(sza)
    Al21[np.abs(ALFA1 - 1.3) > 1e-3] = 0.009649 + 0.005536 * ALFA1[np.abs(ALFA1 - 1.3) > 1e-3] - 0.009349 * ALFA12[np.abs(ALFA1 - 1.3) > 1e-3]

    Al02 = 1.0534924 * np.ones_like(sza)
    Al02[np.abs(ALFA2 - 1.3) > 1e-3] = 1.128036 - 0.0642 * ALFA2[np.abs(ALFA2 - 1.3) > 1e-3] + 0.005276 * ALFA22[np.abs(ALFA2 - 1.3) > 1e-3]
    Al12 = 0.02265614 * np.ones_like(sza)
    Al12[np.abs(ALFA2 - 1.3) > 1e-3] = -0.032851 + 0.036112 * ALFA2[np.abs(ALFA2 - 1.3) > 1e-3] + 0.005066 * ALFA22[np.abs(ALFA2 - 1.3) > 1e-3]
    Al22 = 0.07569785 * np.ones_like(sza)
    Al22[np.abs(ALFA2 - 1.3) > 1e-3] = 0.027787 + 0.064655 * ALFA2[np.abs(ALFA2 - 1.3) > 1e-3] - 0.021385 * ALFA22[np.abs(ALFA2 - 1.3) > 1e-3]

    # transformed optical masses
    X1 = np.log(1 + AMp)
    X2 = np.log(1 + AMdp)
    XX1 = AMp
    XX2 = AMdp
    Y1 = AMW
    Y2 = AMd
    YY1 = np.log(AMp)
    YY2 = np.log(AMdp)

    # when W >= 0.001
    YW1 = np.nan * np.ones_like(sza)
    YW1[W >= 0.001] = np.log(AMW[W >= 0.001] * W[W >= 0.001])
    YW2 = np.nan * np.ones_like(sza)
    YW2[W >= 0.001] = np.log(AMd * W[W >= 0.001])

    # turbidity and other functions
    BETAM1 = AMW * BETA
    BETAM2 = AMd * BETA

    S11 = 1 - np.exp(-0.253565 + 0.061685 * X1 - 0.003997 * X1 * X1)
    TR1 = 0.999523 - 0.010274 * XX1 + 7.375e-5 * XX1 * XX1
    AMG1 = 0.0224 + 0.0094 * YY1 + 0.0019 * YY1 * YY1
    ALM1 = np.log(1 + BETAM1)

    S12 = 1 - np.exp(-0.253565 + 0.061685 * X2 - 0.003997 * X2 * X2)
    TR2 = 0.999523 - 0.010274 * XX2 + 7.375e-5 * XX2 * XX2
    AMG2 = 0.0224 + 0.0094 * YY2 + 0.0019 * YY2 * YY2
    ALM2 = np.log(1 + BETAM2)

    # when W >= 0.001
    AWV1 = np.nan * np.ones_like(sza)
    AWV1[W >= 0.001] = 0.1779 + 0.0519 * YW1[W >= 0.001] + 0.0033 * YW1[W >= 0.001] * YW1[W >= 0.001]
    AWV2 = np.nan * np.ones_like(sza)
    AWV2[W >= 0.001] = 0.1779 + 0.0519 * YW2[W >= 0.001] + 0.0033 * YW2[W >= 0.001] * YW2[W >= 0.001]

    # equivalent wavelengths
    WVLTH11 = Al01 + Al11 * ALM1 + Al21 * ALM1 * ALM1
    WVLTH12 = Al01 + Al11 * ALM2 + Al21 * ALM2 * ALM2
    WVLTH21 = Al02 + Al12 * ALM1 + Al22 * ALM1 * ALM1
    WVLTH22 = Al02 + Al12 * ALM2 + Al22 * ALM2 * ALM2

    # transmittances
    ALFAZ1 = np.zeros_like(sza)
    ALFAZ1[Uo > 0] = np.exp(-2.5686 + 0.6706 * np.log(Uo[Uo > 0] * AMZ[Uo > 0]))

    TAUZ1 = 1 - ALFAZ1
    TAUZ = 1 - ALFAZ1 * F1 / F12
    TAUR1 = np.exp(-AMp * S11)
    TAURM1 = np.exp(-AMdp * S12)
    TAUR2 = TR1
    TAURM2 = TR2
    TAUR = (F1 * TAUR1 + F2 * TAUR2) / F12

    # when W >= 0.001
    TAUW2 = np.ones_like(sza)
    TAUW2[W >= 0.001] = 1 - AWV1[W >= 0.001]
    TAUWP2 = np.ones_like(sza)
    TAUWP2[W >= 0.001] = 1 - AWV2[W >= 0.001]

    TAUW = 1 - AWV1 * F2 / F12
    TAUG2 = 1
    TAUGP2 = 1
    TAUG2 = 1 - AMG1
    TAUGP2 = 1 - AMG2
    TAUG = 1 - AMG1 * F2 / F12

    # aerosol directional factor
    BA = 1 - np.exp(-0.6931 - 1.8326 * np.cos(sza))
    BAM = 0.8342 * np.ones_like(sza)
    BA[BETA <= 0] = 1
    BAM[BETA <= 0] = 1

    # aerosol transmittances and albedos for the two bands
    TAUA1 = np.ones_like(sza)
    TAUA2 = np.ones_like(sza)
    TAUD1 = np.ones_like(sza)
    TAUD2 = np.ones_like(sza)
    TMD1 = np.ones_like(sza)
    TMD2 = np.ones_like(sza)
    TAUM1 = np.ones_like(sza)
    TAUM2 = np.ones_like(sza)

    TAUA1[BETA > 0] = np.maximum(np.exp(-BETAM1[BETA > 0] / (WVLTH11[BETA > 0] ** ALFA1[BETA > 0])), 0.0001)
    TAUA1[TAUA1 >= 1] = 1
    SIGMA1 = -np.log(TAUA1) / AMW
    TAUD1[BETA > 0] = np.exp(-AMW[BETA > 0] * SIGMA1[BETA > 0] * PIAR1)
    TMD1[BETA > 0] = np.minimum(np.exp(-AMd * SIGMA1[BETA > 0] * PIAR1), 1)
    TAUM1[BETA > 0] = np.exp(-BETAM2[BETA > 0] / (WVLTH12[BETA > 0] ** ALFA1[BETA > 0]))
    TAUM2[BETA > 0] = np.exp(-BETAM2[BETA > 0] / (WVLTH22[BETA > 0] ** ALFA2[BETA > 0]))
    TAUA2[BETA > 0] = np.maximum(np.exp(-BETAM1[BETA > 0] / (WVLTH21[BETA > 0] ** ALFA2[BETA > 0])), 0.0001)
    TAUA2[TAUA2 >= 1] = 1
    SIGMA2 = -np.log(TAUA2) / AMW
    TMD2[BETA > 0] = np.minimum(np.exp(-AMd * SIGMA2[BETA > 0] * PIAR2), 1)
    TAUD2[BETA > 0] = np.exp(-AMW[BETA > 0] * SIGMA2[BETA > 0] * PIAR2)

    TAUAA1 = TAUA1 / TAUD1
    TAUAA2 = TAUA2 / TAUD2
    TMA1 = np.minimum(TAUM1 / TMD1, 1)
    TMA2 = np.minimum(TAUM2 / TMD2, 1)

    # sky albedo
    RO1A = (1 - TMD1) * (1 - BAM) * TMA1 * TAURM1
    ROR1 = 0.5 * (1 - TAURM1) * TMA1
    RO1 = RO1A + ROR1
    RO2 = (1 - TMD2) * (1 - BAM) * TMA2 * TAURM2 * TAUWP2 * TAUGP2
    TAUA = (F1 * TAUA1 + F2 * TAUA2) / F12

    # band 1: visible part of the spectrum
    ESC1 = En0 * F1
    EHOR1 = ESC1 * np.cos(sza)
    TAU1 = TAUZ1 * TAUR1
    ENORM1 = TAU1 * TAUA1 * ESC1
    EDIFR1 = 0.5 * TAUZ1 * (1 - TAUR1) * TAUAA1 * EHOR1
    EDIFA1 = BA * TAU1 * (1 - TAUD1) * TAUAA1 * EHOR1
    EDIF10 = EDIFR1 + EDIFA1
    EDIR1 = ENORM1 * np.cos(sza)
    EDIFD1 = RO1 * ROG * (EDIR1 + EDIF10) / (1 - RO1 * ROG)
    EDIF1 = EDIF10 + EDIFD1
    ETOT1 = EDIF1 + EDIR1

    # band 2: infrared part of the spectrum
    ESC2 = En0 * F2
    EHOR2 = ESC2 * np.cos(sza)
    TAU2 = TAUR2 * TAUW2 * TAUG2
    ENORM2 = TAU2 * TAUA2 * ESC2
    EDIFR2 = 0.5 * (1 - TAUR2) * TAUAA2 * TAUW2 * TAUG2 * EHOR2
    EDIFA2 = BA * TAU2 * (1 - TAUD2) * TAUAA2 * EHOR2
    EDIF20 = EDIFR2 + EDIFA2
    EDIR2 = ENORM2 * np.cos(sza)
    EDIFD2 = RO2 * ROG * (EDIR2 + EDIF20) / (1 - RO2 * ROG)
    EDIF2 = EDIF20 + EDIFD2
    ETOT2 = EDIF2 + EDIR2

    # whole spectrum results
    EbnCPCR2 = ENORM1 + ENORM2
    EdhCPCR2 = EDIF1 + EDIF2
    EghCPCR2 = ETOT1 + ETOT2

    # Quality control
    lower = 0
    EbnCPCR2[EbnCPCR2 < lower] = lower
    EdhCPCR2[EdhCPCR2 < lower] = lower
    EghCPCR2[EghCPCR2 < lower] = lower

    return [EghCPCR2, EbnCPCR2, EdhCPCR2]