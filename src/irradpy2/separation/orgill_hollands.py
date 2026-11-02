"""Orgill-Hollands irradiance separation model."""

import numpy as np
import pandas as pd


def _prepare(values, n_rows, name):
    """Convert input values to an (N, 1) float array."""
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        if len(values) != n_rows:
            raise ValueError(f"{name} length must match datetime length.")
        values = values.reshape(-1, 1)
    elif values.ndim == 2:
        if values.shape != (n_rows, 1):
            raise ValueError(f"{name} must have shape (N,) or (N, 1).")
    else:
        raise ValueError(f"{name} must have shape (N,) or (N, 1).")

    return values


def _time(datetime):
    """Parse datetime values as UTC."""
    datetime = pd.DatetimeIndex(pd.to_datetime(np.asarray(datetime).reshape(-1), errors="coerce", utc=True))

    if len(datetime) == 0:
        raise ValueError("datetime cannot be empty.")
    if datetime.isna().any():
        raise ValueError("datetime contains invalid values.")

    return datetime


def _solar(datetime, lat, lon):
    """Calculate day of year and solar zenith angle."""
    day_of_year = np.asarray(datetime.dayofyear, dtype=float).reshape(-1, 1)
    decimal_time = (np.asarray(datetime.hour, dtype=float) + np.asarray(datetime.minute, dtype=float) / 60 + np.asarray(datetime.second, dtype=float) / 3600).reshape(-1, 1)

    beta_eot = 360 / 365.242 * (day_of_year - 1)
    eot = 0.258 * np.cos(np.deg2rad(beta_eot)) - 7.416 * np.sin(np.deg2rad(beta_eot)) - 3.648 * np.cos(np.deg2rad(2 * beta_eot)) - 9.228 * np.sin(np.deg2rad(2 * beta_eot))

    solar_noon = 12 - lon / 15 - eot / 60
    hour_angle = (decimal_time - solar_noon) * 15
    hour_angle = (hour_angle + 180) % 360 - 180

    phi = 2 * np.pi / 365 * (day_of_year + decimal_time / 24 - 1)
    declination = 0.006918 - 0.399912 * np.cos(phi) + 0.070257 * np.sin(phi) - 0.006758 * np.cos(2 * phi) + 0.000907 * np.sin(2 * phi) - 0.002697 * np.cos(3 * phi) + 0.001480 * np.sin(3 * phi)

    cos_zen = np.sin(np.deg2rad(lat)) * np.sin(declination) + np.cos(np.deg2rad(lat)) * np.cos(declination) * np.cos(np.deg2rad(hour_angle))
    zen = np.arccos(np.clip(cos_zen, -1, 1))

    return day_of_year, zen


def _extra(day_of_year):
    """Calculate extraterrestrial normal irradiance."""
    beta = 2 * np.pi * (day_of_year - 1) / 365
    return 1366.1 * (1.00011 + 0.034221 * np.cos(beta) + 0.00128 * np.sin(beta) + 0.000719 * np.cos(2 * beta) + 0.000077 * np.sin(2 * beta))


def orgill_hollands(ghi: np.ndarray, datetime: np.ndarray, lat: float, lon: float):
    """
    Orgill-Hollands irradiance separation model.

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

    Returns
    -------
    kd : numpy.ndarray
        Diffuse fraction with shape (N, 1).
    dhi : numpy.ndarray
        Diffuse horizontal irradiance [W/m^2] with shape (N, 1).
    dni : numpy.ndarray
        Direct normal irradiance [W/m^2] with shape (N, 1).

    References
    ----------
    Orgill, J. F., and Hollands, K. G. T. (1977).
    Correlation equation for hourly diffuse radiation on a horizontal
    surface. Solar Energy, 19(4), 357-359.
    """
    lat = float(lat)
    lon = float(lon)

    if not -90 <= lat <= 90:
        raise ValueError("lat must be within [-90, 90].")
    if not -180 <= lon <= 180:
        raise ValueError("lon must be within [-180, 180].")

    datetime = _time(datetime)
    ghi = _prepare(ghi, len(datetime), "ghi")

    day_of_year, zen = _solar(datetime, lat, lon)
    e_extn = _extra(day_of_year)
    cos_zen = np.cos(zen)

    e_exth = e_extn * cos_zen
    e_exth[zen > np.deg2rad(90)] = 0

    kt = np.divide(ghi, e_exth, out=np.full_like(ghi, np.nan), where=e_exth != 0)
    kt = np.clip(kt, 0, 1)

    kd = 1 - 0.249 * kt
    kd = np.where((kt >= 0.35) & (kt <= 0.75), 1.557 - 1.84 * kt, kd)
    kd = np.where(kt > 0.75, 0.177, kd)
    kd = np.clip(kd, 0, 1)

    dhi = ghi * kd

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