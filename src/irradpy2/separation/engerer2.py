"""Engerer2 irradiance separation model."""

import numpy as np
import pandas as pd


def _prepare_column_input(values, n_rows):
    """
    Keep compatibility with the original Engerer2 implementation.
    Only fix shape issues to avoid NumPy broadcasting explosions.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim == 0:
        values = np.full((n_rows, 1), values)
    elif values.ndim == 1:
        if len(values) == 1:
            values = np.full((n_rows, 1), values[0])
        elif len(values) == n_rows:
            values = values.reshape(-1, 1)
        else:
            raise ValueError("Input length must be 1 or match the number of rows.")
    elif values.ndim == 2:
        if values.shape == (1, 1):
            values = np.full((n_rows, 1), values[0, 0])
        elif values.shape[1] == 1 and values.shape[0] == n_rows:
            pass
        else:
            raise ValueError("Input must be a column vector with shape (N, 1).")
    else:
        raise ValueError("Unsupported input shape.")

    return values


def _parse_datetime(datetime):
    """Parse datetime values as UTC."""
    datetime = pd.DatetimeIndex(pd.to_datetime(np.asarray(datetime).reshape(-1), errors="coerce", utc=True))

    if len(datetime) == 0:
        raise ValueError("datetime cannot be empty.")
    if datetime.isna().any():
        raise ValueError("datetime contains invalid values.")

    return datetime


def _extract_datetime_data(datetime):
    hour = np.asarray(datetime.hour, dtype=float)
    minute = np.asarray(datetime.minute, dtype=float)
    second = np.asarray(datetime.second, dtype=float)
    day_of_year = np.asarray(datetime.dayofyear, dtype=float)

    return hour, minute, second, day_of_year


def _compute_decimal_time(hour, minute, second):
    return hour + minute / 60 + second / 3600


def _compute_day_of_year_and_decimal_time(ghi, day_of_year, decimal_time):
    day_of_year = np.asarray(day_of_year, dtype=float).reshape(len(ghi), 1)
    decimal_time = np.asarray(decimal_time, dtype=float).reshape(len(ghi), 1)

    return day_of_year, decimal_time


def _compute_engerer2_solar_terms(day_of_year, decimal_time, lat, lon):
    """Preserve the original Engerer2 solar geometry formulas."""
    beta_eot = (360 / 365.242) * (day_of_year - 1)
    eot = 0.258 * np.cos(np.pi / 180 * beta_eot) - 7.416 * np.sin(np.pi / 180 * beta_eot) - 3.648 * np.cos(np.pi / 180 * 2 * beta_eot) - 9.228 * np.sin(np.pi / 180 * 2 * beta_eot)

    lsn = 12 - lon / 15 - eot / 60

    hour_angle = (decimal_time - lsn) * 15
    hour_angle[hour_angle >= 180] = hour_angle[hour_angle >= 180] - 360
    hour_angle[hour_angle <= -180] = 360 + hour_angle[hour_angle <= -180]

    phi_r = (2 * np.pi / 365) * (day_of_year + decimal_time / 24 - 1)
    delta_r = 0.006918 - 0.399912 * np.cos(phi_r) + 0.070257 * np.sin(phi_r) - 0.006758 * np.cos(2 * phi_r) + 0.000907 * np.sin(2 * phi_r) - 0.002697 * np.cos(3 * phi_r) + 0.001480 * np.sin(3 * phi_r)

    cos_zen = np.sin(np.pi / 180 * lat) * np.sin(delta_r) + np.cos(np.pi / 180 * lat) * np.cos(delta_r) * np.cos(np.pi / 180 * hour_angle)
    zen = np.arccos(np.clip(cos_zen, -1, 1))

    return eot, lsn, hour_angle, delta_r, zen


def _compute_extraterrestrial_irradiance(day_of_year, zen):
    """Preserve the original extraterrestrial irradiance formula."""
    solar_constant = 1366.1
    beta = (2 * np.pi * day_of_year) / 365
    e_extn = solar_constant * (1.00011 + 0.034221 * np.cos(beta) + 0.00128 * np.sin(beta) + 0.000719 * np.cos(2 * beta) + 0.000077 * np.sin(2 * beta))

    e_exth = e_extn * np.cos(zen)
    e_exth[zen > np.pi / 180 * 90] = 0

    return e_extn, e_exth


def _compute_tj_clear_sky_reference(day_of_year, zen):
    """Preserve the original TJ clear-sky model."""
    a_tj = 1160 + 75 * np.sin(np.deg2rad((360 * (day_of_year - 275)) / 365))
    k_tj = 0.174 + 0.035 * np.sin(np.deg2rad((360 * (day_of_year - 100)) / 365))
    c_tj = 0.095 + 0.04 * np.sin(np.deg2rad((360 * (day_of_year - 100)) / 365))

    cos_zen = np.cos(zen)
    valid = cos_zen > 0

    dnics = np.zeros_like(zen, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dnics[valid] = a_tj[valid] * np.exp(-k_tj[valid] / cos_zen[valid])

    ghics = dnics * cos_zen + c_tj * dnics
    dnics[~valid] = 0
    ghics[~valid] = 0

    return dnics, ghics


def _engerer2_formula(c, b0, b1, b2, b3, b4, b5, zen, ast, dktc, k_de, kt):
    x = b0 + b1 * kt + b2 * ast + b3 * zen / (np.pi / 180) + b4 * dktc

    with np.errstate(over="ignore", invalid="ignore"):
        return c + (1 - c) / (1 + np.exp(x)) + b5 * k_de


def engerer2(ghi: np.ndarray, datetime: np.ndarray, lat: float, lon: float, averaging_period: int):
    """
    Engerer2 irradiance separation model.

    Parameters
    ----------
    ghi : numpy.ndarray
        Global horizontal irradiance [W/m^2], with shape (N,) or (N, 1).
    datetime : numpy.ndarray
        UTC datetime values with length N.
    lat : float
        Site latitude [degrees].
    lon : float
        Site longitude [degrees].
    averaging_period : int
        Averaging period in minutes. Supported values are 1, 5, 10, 15, 30,
        60, and 1440.

    Returns
    -------
    kd : numpy.ndarray
        Diffuse fraction with shape (N, 1).
    dhi : numpy.ndarray
        Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
    dni : numpy.ndarray
        Direct normal irradiance [W/m^2] with shape (N, 1).
    """
    lat = float(lat)
    lon = float(lon)

    if not -90 <= lat <= 90:
        raise ValueError("lat must be within [-90, 90].")
    if not -180 <= lon <= 180:
        raise ValueError("lon must be within [-180, 180].")

    datetime = _parse_datetime(datetime)
    n_rows = len(datetime)
    ghi = _prepare_column_input(ghi, n_rows)

    hour, minute, second, day_of_year_value = _extract_datetime_data(datetime)
    decimal_time_value = _compute_decimal_time(hour, minute, second)
    day_of_year, decimal_time = _compute_day_of_year_and_decimal_time(ghi, day_of_year_value, decimal_time_value)

    eot, lsn, hour_angle, delta_r, zen = _compute_engerer2_solar_terms(day_of_year, decimal_time, lat, lon)
    e_extn, e_exth = _compute_extraterrestrial_irradiance(day_of_year, zen)
    dnics, ghics = _compute_tj_clear_sky_reference(day_of_year, zen)

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi / e_exth
        ktc = ghics / e_exth
        dktc = ktc - kt

    ast = hour_angle / 15 + 12
    ast[ast < 0] = abs(ast[ast < 0])

    cloud_enhancement = ghi - ghics
    cloud_enhancement[cloud_enhancement < 0.015] = 0

    with np.errstate(divide="ignore", invalid="ignore"):
        k_de = cloud_enhancement / ghi

    kt[~np.isfinite(kt)] = np.nan
    ktc[~np.isfinite(ktc)] = np.nan
    dktc[~np.isfinite(dktc)] = np.nan
    ast[~np.isfinite(ast)] = np.nan
    k_de[~np.isfinite(k_de)] = np.nan

    invalid = zen > np.pi / 180 * 90
    kt[invalid] = np.nan
    ktc[invalid] = np.nan
    dktc[invalid] = np.nan
    ast[invalid] = np.nan
    k_de[invalid] = np.nan

    if averaging_period == 1:
        parameters = (0.105620, -4.13320, 8.25780, 0.0100870, 0.000888010, -4.93020, 0.443780)
    elif averaging_period == 5:
        parameters = (0.0939360, -4.57710, 8.46410, 0.0100120, 0.003975, -4.39210, 0.393310)
    elif averaging_period == 10:
        parameters = (0.0799650, -4.85390, 8.47640, 0.0188490, 0.00514970, -4.14570, 0.374660)
    elif averaging_period == 15:
        parameters = (0.0659720, -4.72110, 8.32940, 0.00954440, 0.00534930, -4.16900, 0.395260)
    elif averaging_period == 30:
        parameters = (0.0326750, -4.86810, 8.18670, 0.0158290, 0.00599220, -4.03040, 0.473710)
    elif averaging_period == 60:
        parameters = (-0.00975390, -5.31690, 8.50840, 0.0132410, 0.00743560, -3.03290, 0.564030)
    elif averaging_period == 1440:
        parameters = (0.327260, -9.43910, 17.1130, 0.137520, -0.0240990, 6.62570, 0.314190)
    else:
        raise ValueError("averaging_period must be 1, 5, 10, 15, 30, 60, or 1440 minutes.")

    c, b0, b1, b2, b3, b4, b5 = parameters
    kd = _engerer2_formula(c, b0, b1, b2, b3, b4, b5, zen, ast, dktc, k_de, kt)

    kd[kd > 1] = 1
    kd[kd < 0] = 0

    dhi = ghi * kd
    cos_zen = np.cos(zen)

    high_zen = zen > np.deg2rad(87)
    valid_dni = (~high_zen) & np.isfinite(ghi) & np.isfinite(dhi) & np.isfinite(cos_zen) & (cos_zen > 0)

    dni = np.full_like(ghi, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dni[valid_dni] = (ghi[valid_dni] - dhi[valid_dni]) / cos_zen[valid_dni]

    safe_high_zen = high_zen & np.isfinite(ghi)
    dni[safe_high_zen] = 0
    dhi[safe_high_zen] = np.maximum(ghi[safe_high_zen], 0)
    kd[safe_high_zen] = np.where(ghi[safe_high_zen] > 0, 1.0, 0.0)

    return kd, dhi, dni