import numpy as np
import pandas as pd


def _prepare_column_input(values, n_rows):
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
    datetime = pd.to_datetime(np.asarray(datetime).reshape(-1), errors="coerce", utc=True)
    datetime = pd.DatetimeIndex(datetime)

    if datetime.isna().any():
        raise ValueError("datetime contains invalid values.")

    return datetime


def _extract_datetime_data(datetime):
    datetime = _parse_datetime(datetime)

    hour = np.asarray(datetime.hour, dtype=float)
    minute = np.asarray(datetime.minute, dtype=float)
    second = np.asarray(datetime.second, dtype=float)
    day_of_year = np.asarray(datetime.dayofyear, dtype=float)

    return hour, minute, second, day_of_year


def _compute_decimal_time(hour, minute, second):
    decimal_time = hour + minute / 60 + second / 3600
    return decimal_time


def _compute_day_of_year_and_decimal_time(ghi, day_of_year, decimal_time):
    day_of_year = np.asarray(day_of_year, dtype=float).reshape(len(ghi), 1)
    decimal_time = np.asarray(decimal_time, dtype=float).reshape(len(ghi), 1)

    return day_of_year, decimal_time


def _compute_solar_terms(day_of_year, decimal_time, lat, lon):
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

    return zen


def _compute_extraterrestrial_irradiance(day_of_year, zen):
    solar_constant = 1366.1
    beta = (2 * np.pi * day_of_year) / 365

    e_extn = solar_constant * (1.00011 + 0.034221 * np.cos(beta) + 0.00128 * np.sin(beta) + 0.000719 * np.cos(2 * beta) + 0.000077 * np.sin(2 * beta))

    e_exth = e_extn * np.cos(zen)
    e_exth[zen > np.pi / 180 * 90] = 0

    return e_extn, e_exth


def erbs(ghi, datetime, lat, lon):
    datetime = _parse_datetime(datetime)
    n_rows = len(datetime)

    ghi = _prepare_column_input(ghi, n_rows)
    lat = _prepare_column_input(lat, n_rows)
    lon = _prepare_column_input(lon, n_rows)

    hour, minute, second, day_of_year_value = _extract_datetime_data(datetime)
    decimal_time_value = _compute_decimal_time(hour, minute, second)

    day_of_year, decimal_time = _compute_day_of_year_and_decimal_time(ghi, day_of_year_value, decimal_time_value)

    zen = _compute_solar_terms(day_of_year, decimal_time, lat, lon)
    e_extn, e_exth = _compute_extraterrestrial_irradiance(day_of_year, zen)

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi / e_exth

    kt[~np.isfinite(kt)] = np.nan
    kt = np.clip(kt, 0, 1)

    kd = 1 - 0.09 * kt
    kd = np.where((kt > 0.22) & (kt <= 0.8), 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4, kd)
    kd = np.where(kt > 0.8, 0.165, kd)

    kd[kd > 1] = 1
    kd[kd < 0] = 0

    dhi = ghi * kd
    cos_zen = np.cos(zen)

    invalid = (zen > np.deg2rad(87)) | ~np.isfinite(zen) | ~np.isfinite(ghi)
    valid = (~invalid) & (cos_zen > 0)

    dni = np.zeros_like(ghi, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        dni[valid] = (ghi[valid] - dhi[valid]) / cos_zen[valid]

    invalid = invalid | ~np.isfinite(dni) | (dni < 0)

    dni[invalid] = 0
    dhi[invalid] = np.maximum(ghi[invalid], 0)
    kd[invalid] = np.where(ghi[invalid] > 0, 1.0, 0.0)

    return kd, dhi, dni