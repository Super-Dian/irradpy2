"""DISC irradiance separation model."""

import numpy as np
import pandas as pd


def _prepare_array(values, n_rows, name):
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


def _parse_time(datetime):
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

    beta_eot = (360 / 365.242) * (day_of_year - 1)
    eot = 0.258 * np.cos(np.deg2rad(beta_eot)) - 7.416 * np.sin(np.deg2rad(beta_eot)) - 3.648 * np.cos(np.deg2rad(2 * beta_eot)) - 9.228 * np.sin(np.deg2rad(2 * beta_eot))

    lsn = 12 - lon / 15 - eot / 60
    hour_angle = (decimal_time - lsn) * 15
    hour_angle = (hour_angle + 180) % 360 - 180

    phi_r = 2 * np.pi / 365 * (day_of_year + decimal_time / 24 - 1)
    delta_r = 0.006918 - 0.399912 * np.cos(phi_r) + 0.070257 * np.sin(phi_r) - 0.006758 * np.cos(2 * phi_r) + 0.000907 * np.sin(2 * phi_r) - 0.002697 * np.cos(3 * phi_r) + 0.001480 * np.sin(3 * phi_r)

    cos_zen = np.sin(np.deg2rad(lat)) * np.sin(delta_r) + np.cos(np.deg2rad(lat)) * np.cos(delta_r) * np.cos(np.deg2rad(hour_angle))
    zen = np.arccos(np.clip(cos_zen, -1, 1))

    return day_of_year, zen


def _extra(day_of_year):
    """Calculate extraterrestrial normal irradiance."""
    beta = 2 * np.pi * (day_of_year - 1) / 365
    return 1370.0 * (1.00011 + 0.034221 * np.cos(beta) + 0.00128 * np.sin(beta) + 0.000719 * np.cos(2 * beta) + 0.000077 * np.sin(2 * beta))


def _airmass(zen, pressure):
    """Calculate pressure-corrected absolute airmass."""
    zen_deg = np.rad2deg(zen)
    airmass = np.full_like(zen, np.nan)
    valid = zen_deg < 90

    airmass[valid] = 1 / (np.cos(zen[valid]) + 0.15 * (93.885 - zen_deg[valid]) ** -1.253)
    airmass = airmass * pressure / 1013.25
    airmass[airmass > 12] = 12

    return airmass


def _kn(kt, airmass):
    """Calculate the DISC direct clearness index."""
    cloudy = kt <= 0.6

    a = np.where(cloudy, 0.512 + kt * (-1.56 + kt * (2.286 - 2.222 * kt)), -5.743 + kt * (21.77 + kt * (-27.49 + 11.56 * kt)))
    b = np.where(cloudy, 0.37 + 0.962 * kt, 41.4 + kt * (-118.5 + kt * (66.05 + 31.9 * kt)))
    c = np.where(cloudy, -0.28 + kt * (0.932 - 2.048 * kt), -47.01 + kt * (184.2 + kt * (-222.0 + 73.81 * kt)))

    delta_kn = a + b * np.exp(c * airmass)
    knc = 0.866 + airmass * (-0.122 + airmass * (0.0121 + airmass * (-0.000653 + 1.4e-05 * airmass)))

    return knc - delta_kn


def disc(ghi: np.ndarray, datetime: np.ndarray, lat: float, lon: float, pressure: np.ndarray):
    """
    DISC irradiance separation model.

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
    pressure : numpy.ndarray
        Atmospheric pressure [hPa], with shape (N,) or (N, 1).

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
    Maxwell, E. L. (1987). A Quasi-Physical Model for Converting Hourly
    Global Horizontal to Direct Normal Insolation. SERI/TR-215-3087.
    """
    lat = float(lat)
    lon = float(lon)

    if not -90 <= lat <= 90:
        raise ValueError("lat must be within [-90, 90].")
    if not -180 <= lon <= 180:
        raise ValueError("lon must be within [-180, 180].")

    datetime = _parse_time(datetime)
    ghi = _prepare_array(ghi, len(datetime), "ghi")
    pressure = _prepare_array(pressure, len(datetime), "pressure")

    if np.any(pressure <= 0):
        raise ValueError("pressure must contain positive values.")

    day_of_year, zen = _solar(datetime, lat, lon)
    e_extn = _extra(day_of_year)
    cos_zen = np.cos(zen)

    denominator = e_extn * np.maximum(cos_zen, 0.065)
    kt = np.divide(ghi, denominator, out=np.full_like(ghi, np.nan), where=denominator != 0)
    kt = np.clip(kt, 0, 1)

    airmass = _airmass(zen, pressure)
    kn = _kn(kt, airmass)

    dni = kn * e_extn
    dhi = ghi - dni * cos_zen

    dni[dni < 0] = 0
    dhi[dhi < 0] = 0

    kd = np.divide(dhi, ghi, out=np.full_like(dhi, np.nan), where=ghi != 0)
    kd = np.clip(kd, 0, 1)

    invalid = zen > np.deg2rad(87)
    kd[invalid] = np.nan
    dhi[invalid] = np.nan
    dni[invalid] = np.nan

    return kd, dhi, dni