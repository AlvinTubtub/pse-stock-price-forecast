from __future__ import annotations

import builtins
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from urllib.error import URLError

import pytest

from config.companies import COMPANIES
from scripts import update_eod
from src.ingestion.cleaner import NumericCleaningError, clean_number, clean_quotation
from src.ingestion.config import IngestionSettings
from src.ingestion.downloader import (
    DownloadResult,
    FetchResponse,
    download_one,
    report_filename,
    report_url,
)
from src.ingestion.merger import RawMergeError, merge_symbol, validate_raw_file
from src.ingestion.parser import (
    ParseBatchResult,
    PdfParseError,
    parse_pdf,
    parse_quotation_line,
)
from src.ingestion.pipeline import resolve_date_range, run_ingestion


def _settings(tmp_path: Path, *, retries: int = 3) -> IngestionSettings:
    return IngestionSettings(
        reports_dir=tmp_path / "pdf_reports",
        raw_dir=tmp_path / "raw",
        request_timeout_seconds=1,
        max_retries=retries,
        backoff_base_seconds=2,
    )


def _quote(day: date, *, close: str = "101.00"):
    return clean_quotation(
        report_date=day,
        issue_name="BANK PH ISLANDS",
        symbol="BPI",
        open_value="100.00",
        high_value="103.00",
        low_value="99.00",
        close_value=close,
        volume_value="123,000",
        traded_value="12,345,000",
    )


def _write_raw(path: Path, *, close: str = "101.0") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "Date,Open,High,Low,Close,Volume\n"
        f"2026-09-10,100.0,103.0,99.0,{close},123000.0\n"
    ).encode()
    path.write_bytes(content)
    return content


def test_url_generation_matches_pse_edge_contract() -> None:
    assert report_url(date(2026, 9, 7)) == (
        "https://documents.pse.com.ph/market_report/September%2007,%202026-EOD.pdf"
    )


def test_pdf_filename_generation() -> None:
    assert report_filename(date(2026, 9, 7)) == "2026-09-07-EOD.pdf"


def test_404_is_unpublished_and_not_failure(tmp_path: Path) -> None:
    status, path, message = download_one(
        date(2026, 9, 6),
        settings=_settings(tmp_path),
        fetcher=lambda _url, _timeout: FetchResponse(404, b""),
        sleep_fn=lambda _seconds: pytest.fail("404 must not retry"),
    )
    assert (status, path, message) == ("unpublished", None, None)


def test_transient_failure_retries_then_downloads(tmp_path: Path) -> None:
    responses: list[object] = [URLError("temporary outage"), FetchResponse(200, b"%PDF-test")]
    waits: list[float] = []

    def fetcher(_url: str, _timeout: float) -> FetchResponse:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    status, path, message = download_one(
        date(2026, 9, 7),
        settings=_settings(tmp_path),
        fetcher=fetcher,
        sleep_fn=waits.append,
    )
    assert status == "downloaded"
    assert message is None
    assert path is not None and path.read_bytes() == b"%PDF-test"
    assert waits == [2]


def test_transient_server_error_retries(tmp_path: Path) -> None:
    responses = [FetchResponse(503, b""), FetchResponse(200, b"%PDF-test")]
    waits: list[float] = []

    status, path, _message = download_one(
        date(2026, 9, 7),
        settings=_settings(tmp_path),
        fetcher=lambda _url, _timeout: responses.pop(0),
        sleep_fn=waits.append,
    )
    assert status == "downloaded"
    assert path is not None
    assert waits == [2]


def test_existing_pdf_is_skipped_without_network_call(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    destination = settings.reports_dir / report_filename(date(2026, 9, 7))
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    def unexpected_fetch(_url: str, _timeout: float) -> FetchResponse:
        pytest.fail("existing report must not be downloaded again")

    status, path, message = download_one(
        date(2026, 9, 7), settings=settings, fetcher=unexpected_fetch
    )
    assert (status, path, message) == ("already_present", destination, None)
    assert destination.read_bytes() == b"existing"


class _Page:
    def __init__(self, text: str | None):
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class _Pdf:
    def __init__(self, pages: list[_Page]):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_parser_extracts_only_configured_company_rows(tmp_path: Path) -> None:
    path = tmp_path / "2026-09-07-EOD.pdf"
    path.write_bytes(b"placeholder")
    text = "\n".join(
        (
            "END OF DAY QUOTATION REPORT September 7, 2026",
            "BANK PH ISLANDS BPI 99 100 100 103 99 101 123,000 12,345,000 1",
            "UNKNOWN COMPANY XYZ 1 2 1 2 1 2 0 5 10",
        )
    )
    records = parse_pdf(path, pdf_opener=lambda _path: _Pdf([_Page(text)]))
    assert len(records) == 1
    record = records[0]
    assert record.symbol == "BPI"
    assert record.issue_name == "BANK PH ISLANDS"
    assert record.open == Decimal("100")
    assert record.close == Decimal("101")
    assert record.volume == Decimal("123000")
    assert record.value == Decimal("12345000")


def test_parser_quotation_layout_matches_actual_pse_columns() -> None:
    record = parse_quotation_line(
        "BANK PH ISLANDS BPI 99 100 100 103 99 101 123,000 12,345,000 1",
        date(2026, 9, 7),
    )
    assert record is not None
    assert (record.open, record.high, record.low, record.close) == (
        Decimal("100"),
        Decimal("103"),
        Decimal("99"),
        Decimal("101"),
    )
    assert record.volume == Decimal("123000")
    assert record.value == Decimal("12345000")


def test_glo_token_inside_unrelated_issue_name_is_not_a_target_row() -> None:
    assert (
        parse_quotation_line(
            "GLO PREF ANV GLOBA 1,925 1,964 - - - - - - -",
            date(2026, 9, 11),
        )
        is None
    )


def test_genuine_glo_common_stock_row_is_parsed_once_and_correctly() -> None:
    record = parse_quotation_line(
        "GLOBE TELECOM GLO 1,598 1,600 1,620 1,623 1,595 1,600 "
        "31,180 49,938,475 (21,146,560)",
        date(2026, 9, 11),
    )
    assert record is not None
    assert record.symbol == "GLO"
    assert record.issue_name == "GLOBE TELECOM"
    assert record.open == Decimal("1620")
    assert record.high == Decimal("1623")
    assert record.low == Decimal("1595")
    assert record.close == Decimal("1600")
    assert record.volume == Decimal("31180")
    assert record.value == Decimal("49938475")


def test_genuine_target_row_with_malformed_ohlcv_remains_strict() -> None:
    with pytest.raises(PdfParseError, match="Malformed quotation row for GLO"):
        parse_quotation_line(
            "GLOBE TELECOM GLO 1,598 1,600 bad 1,623 1,595 1,600 "
            "31,180 49,938,475 (21,146,560)",
            date(2026, 9, 11),
        )


def test_false_positive_line_cannot_invalidate_full_configured_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2026-09-11-EOD.pdf"
    path.write_bytes(b"placeholder")
    quote_lines = [
        f"{company.pse_issue_name} {company.symbol} "
        "99 100 100 103 99 101 123,000 12,345,000 1"
        for company in COMPANIES
    ]
    text = "\n".join(
        [
            "Daily Quotation Report September 11, 2026",
            "GLO PREF ANV GLOBA 1,925 1,964 - - - - - - -",
            *quote_lines,
        ]
    )
    records = parse_pdf(path, pdf_opener=lambda _path: _Pdf([_Page(text)]))
    assert len(records) == len(COMPANIES) == 15
    assert {record.symbol for record in records} == {
        company.symbol for company in COMPANIES
    }
    assert sum(record.symbol == "GLO" for record in records) == 1


def test_numeric_cleaning_is_deterministic_and_strict() -> None:
    assert clean_number(" 1,234.50 ") == Decimal("1234.50")
    assert clean_number("(2,500)") == Decimal("-2500")
    assert clean_number("-") is None
    with pytest.raises(NumericCleaningError, match="Malformed numeric"):
        clean_number("1O0.00")


def test_duplicate_incoming_date_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RawMergeError, match="Duplicate incoming date"):
        merge_symbol(
            "BPI",
            [_quote(date(2026, 9, 11)), _quote(date(2026, 9, 11))],
            raw_dir=tmp_path / "raw",
        )


def test_idempotent_merge_does_not_rewrite_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "BPI.csv"
    original = _write_raw(path)
    summary = merge_symbol(
        "BPI", [_quote(date(2026, 9, 10))], raw_dir=path.parent
    )
    assert summary.rows_added == 0
    assert path.read_bytes() == original


def test_conflicting_historical_row_is_rejected_without_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "BPI.csv"
    original = _write_raw(path)
    with pytest.raises(RawMergeError, match="Conflicting historical row"):
        merge_symbol(
            "BPI",
            [_quote(date(2026, 9, 10), close="102.00")],
            raw_dir=path.parent,
        )
    assert path.read_bytes() == original


def test_merge_runs_authoritative_raw_validation(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "BPI.csv"
    _write_raw(path)
    summary = merge_symbol(
        "BPI", [_quote(date(2026, 9, 11))], raw_dir=path.parent
    )
    records = validate_raw_file(summary.path)
    assert [record.trading_date for record in records] == [
        date(2026, 9, 10),
        date(2026, 9, 11),
    ]
    assert summary.rows_added == 1


def test_default_range_starts_after_latest_raw_date(tmp_path: Path) -> None:
    _write_raw(tmp_path / "raw" / "BPI.csv")
    start, end = resolve_date_range(
        raw_dir=tmp_path / "raw",
        start_date=None,
        end_date=None,
        now=datetime.fromisoformat("2026-09-12T00:30:00+08:00"),
    )
    assert start == date(2026, 9, 11)
    assert end == date(2026, 9, 12)


def test_ingestion_has_no_training_or_forecast_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guarded_prefixes = (
        "src.training",
        "src.models",
        "src.evaluation",
        "src.inference",
        "src.export",
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(guarded_prefixes):
            pytest.fail(f"ingestion imported out-of-scope module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    def all_unpublished(_start, _end, *, settings):
        return DownloadResult(unpublished=[date(2026, 9, 6)])

    result = run_ingestion(
        start_date=date(2026, 9, 6),
        end_date=date(2026, 9, 6),
        settings=_settings(tmp_path),
        download_function=all_unpublished,
    )
    assert result.successful
    assert result.validation_status == "not_required_no_published_reports"
    assert not (tmp_path / "artifacts").exists()


def test_pipeline_validates_raw_after_merge(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = settings.reports_dir / report_filename(date(2026, 9, 11))
    report.parent.mkdir(parents=True)
    report.write_bytes(b"placeholder")

    def downloaded(_start, _end, *, settings):
        return DownloadResult(already_present=[report])

    def parsed(_paths, *, settings):
        return ParseBatchResult(
            records=[_quote(date(2026, 9, 11))], parsed_reports=[report]
        )

    result = run_ingestion(
        start_date=date(2026, 9, 11),
        end_date=date(2026, 9, 11),
        settings=settings,
        download_function=downloaded,
        parse_function=parsed,
    )
    assert result.successful
    assert result.validation_status == "passed"
    assert result.rows_added == 1
    assert validate_raw_file(settings.raw_dir / "BPI.csv")[-1].trading_date == date(
        2026, 9, 11
    )


def test_cli_summary_and_nonzero_failure_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failure = update_eod.IngestionResult(
        requested_start_date=date(2026, 9, 7),
        requested_end_date=date(2026, 9, 7),
        failed_downloads=[(date(2026, 9, 7), "HTTP 503")],
        validation_status="not_required_no_published_reports",
        errors=["One or more requested reports failed to download"],
    )
    monkeypatch.setattr(update_eod, "run_ingestion", lambda **_kwargs: failure)
    assert update_eod.main(["--start-date", "2026-09-07", "--end-date", "2026-09-07"]) == 1
    output = capsys.readouterr().out
    assert "Requested date range: 2026-09-07 to 2026-09-07" in output
    assert "Failed downloads (1): 2026-09-07: HTTP 503" in output
    assert "Validation status: not_required_no_published_reports" in output
