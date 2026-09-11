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
    pse_issue_name: str

    @property
    def raw_filename(self) -> str:
        return f"{self.symbol}.csv"


COMPANIES: Final[tuple[Company, ...]] = (
    Company("ALI", "Ayala Land, Inc.", "Property", "AYALA LAND"),
    Company("APX", "Apex Mining Co., Inc.", "Mining and Oil", "APEX MINING"),
    Company("BPI", "Bank of the Philippine Islands", "Financials", "BANK PH ISLANDS"),
    Company("GLO", "Globe Telecom, Inc.", "Services", "GLOBE TELECOM"),
    Company(
        "ICT",
        "International Container Terminal Services, Inc.",
        "Services",
        "INTL CONTAINER",
    ),
    Company("JFC", "Jollibee Foods Corporation", "Industrial", "JOLLIBEE"),
    Company("MBT", "Metropolitan Bank & Trust Company", "Financials", "METROBANK"),
    Company("MEG", "Megaworld Corporation", "Property", "MEGAWORLD"),
    Company("MER", "Manila Electric Company", "Industrial", "MERALCO"),
    Company("NIKL", "Nickel Asia Corporation", "Mining and Oil", "NICKEL ASIA"),
    Company("PGOLD", "Puregold Price Club, Inc.", "Services", "PUREGOLD"),
    Company(
        "SCC",
        "Semirara Mining and Power Corporation",
        "Mining and Oil",
        "SEMIRARA MINING",
    ),
    Company("SECB", "Security Bank Corporation", "Financials", "SECURITY BANK"),
    Company("SHLPH", "Shell Pilipinas Corporation", "Industrial", "SHELL PILIPINAS"),
    Company("SMPH", "SM Prime Holdings, Inc.", "Property", "SM PRIME HLDG"),
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
