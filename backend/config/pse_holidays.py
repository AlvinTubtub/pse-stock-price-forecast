"""Reviewed PSE closure dates; no network access or generic holiday generator.

2026 baseline: SCCP memorandum 01-0925 (2025-09-05), regular and special
NON-WORKING days only:
https://www.sccp.com.ph/resources/files/memos/2025/01-0925%20Philippine%20Holidays%20for%202026.pdf
PSE's published trading rule excludes legal holidays and clearing-office closures:
https://www.pse.com.ph/investing-at-pse/
April dates additionally confirmed by PSE CN-2026-0011:
https://documents.pse.com.ph/CircularOPSPDF/CN-2026-0011.pdf
Eid closures: PSE archive notice dated March 13 (March 20 closure), and
CN-2026-0023 dated May 22 (May 27 closure):
https://www.pse.com.ph/news-and-announcement-archive/
https://documents.pse.com.ph/CircularOPSPDF/CN-2026-0023.pdf

Maintain this explicit list from PSE/SCCP notices before each calendar year and
when exceptional closures are announced. February 25 is a WORKING holiday and
is deliberately excluded. Eid dates use separate exchange confirmation;
they are not inferred from a religious/public-holiday package. CLI --holiday
adds emergency dates without replacing this baseline. No closure is inferred
from missing raw observations. This list is not a perpetual exchange calendar.
"""

from datetime import date


PSE_CLOSURE_ISO_DATES = (
    "2026-01-01",
    "2026-02-17",
    "2026-03-20",
    "2026-04-02",
    "2026-04-03",
    "2026-04-04",
    "2026-04-09",
    "2026-05-01",
    "2026-05-27",
    "2026-06-12",
    "2026-08-21",
    "2026-08-31",
    "2026-11-01",
    "2026-11-02",
    "2026-11-30",
    "2026-12-08",
    "2026-12-24",
    "2026-12-25",
    "2026-12-30",
    "2026-12-31",
)

PSE_CLOSURES = frozenset(date.fromisoformat(value) for value in PSE_CLOSURE_ISO_DATES)
