"""Conflict-safe, idempotent merge of EOD quotes into canonical raw CSVs."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
from datetime import date
import io
import logging
import os
from pathlib import Path

from src.data.validator import (
    OhlcvRecord,
    OhlcvValidationError,
    parse_ohlcv_record,
    require_chronological_records,
    validate_and_sort_records,
    validate_required_columns,
)
from src.ingestion.cleaner import QuotationRecord
from src.ingestion.validator import validate_extracted_quotes


LOGGER = logging.getLogger(__name__)
RAW_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


class RawMergeError(ValueError):
    """Raised when a raw file is invalid or an incoming row conflicts with history."""


@dataclass(frozen=True, slots=True)
class MergeSummary:
    symbol: str
    path: Path
    rows_before: int
    rows_after: int
    rows_added: int
    latest_date: date | None


@dataclass(frozen=True, slots=True)
class _ExistingRaw:
    records: tuple[OhlcvRecord, ...]
    header_line: str
    row_lines: dict[date, str]


def validate_raw_file(path: Path) -> tuple[OhlcvRecord, ...]:
    """Run the authoritative new-backend validator against a raw CSV path."""

    with Path(path).open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        validate_required_columns(reader.fieldnames)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise OhlcvValidationError(
                f"Raw CSV columns must be exactly {', '.join(RAW_COLUMNS)}"
            )
        mappings = list(reader)
    records = validate_and_sort_records(mappings)
    input_order = tuple(
        parse_ohlcv_record(row, row_number=index)
        for index, row in enumerate(mappings, start=2)
    )
    require_chronological_records(input_order)
    return records


def _read_existing(path: Path) -> _ExistingRaw:
    records = validate_raw_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(records) + 1:
        raise RawMergeError(f"Unsupported multiline or blank-row CSV layout: {path}")
    return _ExistingRaw(
        records=records,
        header_line=lines[0],
        row_lines={record.trading_date: line for record, line in zip(records, lines[1:])},
    )


def _quote_as_ohlcv(record: QuotationRecord) -> OhlcvRecord:
    return parse_ohlcv_record(record.ohlcv_mapping(), row_number=2)


def _same_values(left: OhlcvRecord, right: OhlcvRecord) -> bool:
    return (
        left.trading_date == right.trading_date
        and left.open == right.open
        and left.high == right.high
        and left.low == right.low
        and left.close == right.close
        and left.volume == right.volume
    )


def _render_quote_line(record: QuotationRecord) -> str:
    target = io.StringIO()
    writer = csv.DictWriter(target, fieldnames=RAW_COLUMNS, lineterminator="\n")
    writer.writerow(record.ohlcv_mapping())
    return target.getvalue().rstrip("\n")


def _validate_temporary_file(path: Path) -> None:
    try:
        validate_raw_file(path)
    except OhlcvValidationError as exc:
        raise RawMergeError(f"Merged raw validation failed for {path.name}: {exc}") from exc


def merge_symbol(
    symbol: str,
    incoming: Sequence[QuotationRecord],
    *,
    raw_dir: Path,
) -> MergeSummary:
    """Merge one symbol, preserving existing row text and rejecting conflicts."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{symbol}.csv"
    if path.exists():
        try:
            existing = _read_existing(path)
        except (OhlcvValidationError, RawMergeError) as exc:
            raise RawMergeError(f"Existing raw data is invalid for {symbol}: {exc}") from exc
    else:
        existing = _ExistingRaw(records=(), header_line=",".join(RAW_COLUMNS), row_lines={})

    existing_by_date = {record.trading_date: record for record in existing.records}
    incoming_by_date: dict[date, QuotationRecord] = {}
    for quote in incoming:
        prior_incoming = incoming_by_date.get(quote.report_date)
        if prior_incoming is not None:
            raise RawMergeError(f"Duplicate incoming date for {symbol}: {quote.report_date}")
        incoming_by_date[quote.report_date] = quote
        prior = existing_by_date.get(quote.report_date)
        if prior is not None and not _same_values(prior, _quote_as_ohlcv(quote)):
            raise RawMergeError(
                f"Conflicting historical row for {symbol} on {quote.report_date}; "
                "stored OHLCV was not overwritten"
            )

    additions = {
        day: quote for day, quote in incoming_by_date.items() if day not in existing_by_date
    }
    before = len(existing.records)
    if not additions:
        latest = existing.records[-1].trading_date if existing.records else None
        LOGGER.info("Raw merge is idempotent symbol=%s rows_added=0", symbol)
        return MergeSummary(symbol, path, before, before, 0, latest)

    row_lines = dict(existing.row_lines)
    row_lines.update({day: _render_quote_line(quote) for day, quote in additions.items()})
    ordered_dates = sorted(row_lines)
    payload = "\n".join(
        [existing.header_line, *(row_lines[day] for day in ordered_dates)]
    ) + "\n"
    temporary = path.with_name(f".{path.name}.ingestion.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="")
        _validate_temporary_file(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    LOGGER.info(
        "Raw merge completed symbol=%s rows_added=%d latest_date=%s",
        symbol,
        len(additions),
        ordered_dates[-1],
    )
    return MergeSummary(
        symbol=symbol,
        path=path,
        rows_before=before,
        rows_after=len(row_lines),
        rows_added=len(additions),
        latest_date=ordered_dates[-1],
    )


def merge_into_raw(
    records: Iterable[QuotationRecord],
    *,
    raw_dir: Path,
) -> tuple[MergeSummary, ...]:
    validated = validate_extracted_quotes(records)
    groups: dict[str, list[QuotationRecord]] = defaultdict(list)
    for record in validated:
        groups[record.symbol].append(record)
    return tuple(
        merge_symbol(symbol, groups[symbol], raw_dir=raw_dir)
        for symbol in sorted(groups)
    )
