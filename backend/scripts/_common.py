"""Small CLI helpers shared by the authoritative backend scripts."""

import argparse
from datetime import date

from config.companies import COMPANIES, get_company


def add_symbol_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--symbol", help="Process one configured PSE symbol")
    selection.add_argument("--all", action="store_true", help="Process all configured symbols")


def selected_symbols(arguments: argparse.Namespace) -> tuple[str, ...]:
    if arguments.all:
        return tuple(company.symbol for company in COMPANIES)
    return (get_company(arguments.symbol).symbol,)


def parse_holiday(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("holiday must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("holiday must use YYYY-MM-DD")
    return parsed


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--holiday",
        action="append",
        default=[],
        type=parse_holiday,
        help="Additional PSE closure date; repeat as needed",
    )
    add_verbose_option(parser)


def add_verbose_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
