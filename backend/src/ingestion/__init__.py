"""PSE EDGE End-of-Day ingestion, isolated from forecasting workflows."""

from src.ingestion.pipeline import IngestionResult, run_ingestion

__all__ = ["IngestionResult", "run_ingestion"]
