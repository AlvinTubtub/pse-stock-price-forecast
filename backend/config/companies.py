"""Single source of truth for the companies modeled by ForecastPH."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


@dataclass(frozen=True, slots=True)
class Company:
    """Configuration for one modeled PSE company."""

    symbol: str
    name: str
    sector: str

    @property
    def raw_filename(self) -> str:
        return f"{self.symbol}.csv"


COMPANIES: Final[tuple[Company, ...]] = (
    Company("ALI", "Ayala Land, Inc.", "Property"),
    Company("APX", "Apex Mining Co., Inc.", "Mining and Oil"),
    Company("BPI", "Bank of the Philippine Islands", "Financials"),
    Company("GLO", "Globe Telecom, Inc.", "Services"),
    Company("ICT", "International Container Terminal Services, Inc.", "Services"),
    Company("JFC", "Jollibee Foods Corporation", "Industrial"),
    Company("MBT", "Metropolitan Bank & Trust Company", "Financials"),
    Company("MEG", "Megaworld Corporation", "Property"),
    Company("MER", "Manila Electric Company", "Industrial"),
    Company("NIKL", "Nickel Asia Corporation", "Mining and Oil"),
    Company("PGOLD", "Puregold Price Club, Inc.", "Services"),
    Company("SCC", "Semirara Mining and Power Corporation", "Mining and Oil"),
    Company("SECB", "Security Bank Corporation", "Financials"),
    Company("SHLPH", "Shell Pilipinas Corporation", "Industrial"),
    Company("SMPH", "SM Prime Holdings, Inc.", "Property"),
)

COMPANY_BY_SYMBOL: Final[Mapping[str, Company]] = MappingProxyType(
    {company.symbol: company for company in COMPANIES}
)


def get_company(symbol: str) -> Company:
    """Return configured company metadata or reject an unknown symbol."""

    normalized = symbol.strip().upper()
    try:
        return COMPANY_BY_SYMBOL[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown configured company symbol: {symbol!r}") from exc
