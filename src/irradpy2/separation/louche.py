"""Louche irradiance separation model."""

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


def louche(ghi: np.ndarray, datetime: np.ndarray, lat: float, lon: float):
    """
    Louche irradiance separation model.

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
    Louche, A., Notton, G., Poggi, P., and Simonnot, G. (1991).
    Correlations for direct normal and global horizontal irradiation on a
    French Mediterranean site. Solar Energy, 46(4), 261-266.
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

    denominator = e_extn * np.maximum(cos_zen, 0.065)
    kt = np.divide(ghi, denominator, out=np.full_like(ghi, np.nan), where=denominator != 0)
    kt = np.clip(kt, 0, 2)

    kb = -10.627 * kt ** 5 + 15.307 * kt ** 4 - 5.205 * kt ** 3 + 0.994 * kt ** 2 - 0.059 * kt + 0.002
    dni = kb * e_extn
    dhi = ghi - dni * cos_zen

    # Handle invalid values while preserving irradiance closure.
    invalid = (zen > np.deg2rad(90)) | (ghi < 0) | (dni < 0)
    dni = np.where(invalid, 0, dni)
    dhi = np.where(invalid, ghi, dhi)

    kd = np.divide(dhi, ghi, out=np.full_like(dhi, np.nan), where=ghi != 0)
    kd[(zen > np.deg2rad(90)) | (ghi < 0)] = np.nan

    return kd, dhi, dni