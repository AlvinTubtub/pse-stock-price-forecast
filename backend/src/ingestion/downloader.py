"""Download official PSE EDGE EOD PDFs with bounded transient retries."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
import logging
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.ingestion.config import DEFAULT_INGESTION_SETTINGS, IngestionSettings, PDF_SUFFIX


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FetchResponse:
    status_code: int
    content: bytes


@dataclass(slots=True)
class DownloadResult:
    downloaded: list[Path] = field(default_factory=list)
    already_present: list[Path] = field(default_factory=list)
    unpublished: list[date] = field(default_factory=list)
    failed: list[tuple[date, str]] = field(default_factory=list)

    @property
    def available_reports(self) -> tuple[Path, ...]:
        return tuple(sorted((*self.downloaded, *self.already_present)))


Fetcher = Callable[[str, float], FetchResponse]


def report_url(report_date: date, *, base_url: str = DEFAULT_INGESTION_SETTINGS.base_url) -> str:
    """Build the PSE EDGE URL used by the main-branch ingestion pipeline."""

    month = report_date.strftime("%B")
    day = report_date.strftime("%d")
    year = report_date.strftime("%Y")
    remote_name = f"{month}%20{day},%20{year}{PDF_SUFFIX}"
    return f"{base_url.rstrip('/')}/{remote_name}"


def report_filename(report_date: date) -> str:
    return f"{report_date.isoformat()}{PDF_SUFFIX}"


def _fetch(url: str, timeout_seconds: float) -> FetchResponse:
    request = Request(url, headers={"User-Agent": "ForecastPH-EOD-Ingestion/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed HTTPS host
        return FetchResponse(status_code=int(response.status), content=response.read())


def _retry_delay(settings: IngestionSettings, attempt: int) -> float:
    return settings.backoff_base_seconds**attempt


def _write_download_atomically(destination: Path, content: bytes) -> None:
    if not content:
        raise OSError("PSE EDGE returned an empty response body")
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_one(
    report_date: date,
    *,
    settings: IngestionSettings = DEFAULT_INGESTION_SETTINGS,
    fetcher: Fetcher = _fetch,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[str, Path | None, str | None]:
    """Download one report and return downloaded/already_present/unpublished/failed."""

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.reports_dir / report_filename(report_date)
    if destination.is_file():
        LOGGER.info("EOD report already present date=%s path=%s", report_date, destination)
        return "already_present", destination, None

    url = report_url(report_date, base_url=settings.base_url)
    last_error = "unknown download failure"
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = fetcher(url, settings.request_timeout_seconds)
            status_code = response.status_code
            content = response.content
        except HTTPError as exc:
            status_code = exc.code
            content = b""
        except (URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "Transient EOD download failure date=%s attempt=%d/%d error=%s",
                report_date,
                attempt,
                settings.max_retries,
                last_error,
            )
            if attempt < settings.max_retries:
                sleep_fn(_retry_delay(settings, attempt))
                continue
            return "failed", None, last_error

        if status_code == 200:
            try:
                _write_download_atomically(destination, content)
            except OSError as exc:
                return "failed", None, f"Could not save report: {exc}"
            LOGGER.info("Downloaded EOD report date=%s bytes=%d", report_date, len(content))
            return "downloaded", destination, None
        if status_code == 404:
            LOGGER.info("EOD report unpublished date=%s", report_date)
            return "unpublished", None, None
        if status_code == 429 or status_code >= 500:
            last_error = f"HTTP {status_code}"
            LOGGER.warning(
                "Transient PSE EDGE response date=%s attempt=%d/%d status=%d",
                report_date,
                attempt,
                settings.max_retries,
                status_code,
            )
            if attempt < settings.max_retries:
                sleep_fn(_retry_delay(settings, attempt))
                continue
            return "failed", None, last_error
        return "failed", None, f"HTTP {status_code}"

    return "failed", None, last_error


def download_reports(
    start_date: date,
    end_date: date,
    *,
    settings: IngestionSettings = DEFAULT_INGESTION_SETTINGS,
    fetcher: Fetcher = _fetch,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Download every calendar date inclusively; 404 dates remain nonfatal."""

    if start_date > end_date:
        return DownloadResult()
    result = DownloadResult()
    current = start_date
    while current <= end_date:
        status, path, message = download_one(
            current,
            settings=settings,
            fetcher=fetcher,
            sleep_fn=sleep_fn,
        )
        if status == "downloaded" and path is not None:
            result.downloaded.append(path)
        elif status == "already_present" and path is not None:
            result.already_present.append(path)
        elif status == "unpublished":
            result.unpublished.append(current)
        else:
            result.failed.append((current, message or "unknown download failure"))
        current += timedelta(days=1)
    return result
