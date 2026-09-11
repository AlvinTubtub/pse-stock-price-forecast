"""Safely clear generated backend artifacts while preserving raw source data."""

import logging

from src.artifacts.manager import reset_generated_artifacts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    directories = reset_generated_artifacts()
    logging.getLogger(__name__).info(
        "Artifact reset complete directories=%s",
        ",".join(path.name for path in directories.all_generated_directories()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
