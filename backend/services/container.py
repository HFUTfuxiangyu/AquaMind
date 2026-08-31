"""Shared service instances used by startup code and API routers."""

from .prediction_service import PredictionService


prediction_service = PredictionService()
