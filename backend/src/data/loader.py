"""Read-only loading of configured company histories from backend/data/raw."""

import csv
import logging
from pathlib import Path
from typing import Final

from config.companies import COMPANIES, Company, get_company
from config.settings import SETTINGS
from src.data.validator import (
    OhlcvRecord,
    validate_and_sort_records,
    validate_required_columns,
)


LOGGER = logging.getLogger(__name__)
RAW_DATA_DIR: Final[Path] = SETTINGS.raw_data_dir.resolve()


def _company_raw_path(company: Company) -> Path:
    path = (RAW_DATA_DIR / company.raw_filename).resolve()
    if path.parent != RAW_DATA_DIR:
        raise ValueError(f"Unsafe raw-data filename for {company.symbol}")
    return path


def load_company_history(symbol: str) -> tuple[OhlcvRecord, ...]:
    """Load one configured company's immutable raw CSV in chronological order."""

    company = get_company(symbol)
    path = _company_raw_path(company)
    LOGGER.info("Loading raw OHLCV symbol=%s path=%s", company.symbol, path)
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        validate_required_columns(reader.fieldnames)
        records = validate_and_sort_records(reader)
    LOGGER.info("Loaded raw OHLCV symbol=%s rows=%d", company.symbol, len(records))
    return records


def load_all_company_histories() -> dict[str, tuple[OhlcvRecord, ...]]:
    """Load every explicitly configured company from the sole raw input directory."""

    LOGGER.info("Loading configured company histories count=%d", len(COMPANIES))
    return {company.symbol: load_company_history(company.symbol) for company in COMPANIES}
