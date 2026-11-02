"""
Solar irradiance forecasting methods.

This module provides classical and deep learning methods for
Global Horizontal Irradiance forecasting.
"""

from .forecast import (
    forecast_persistence,
    forecast_arima,
    forecast_rnn,
    forecast_lstm,
    forecast_informer,
)

__all__ = [
    "forecast_persistence",
    "forecast_arima",
    "forecast_rnn",
    "forecast_lstm",
    "forecast_informer",
]