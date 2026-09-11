"""Causal OHLCV-derived features for Lag-Informed Regression."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
import logging
import math

import numpy as np
from numpy.typing import NDArray

from config.model_config import RegressionFeatureConfig
from src.data.validator import OhlcvRecord, require_chronological_records


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


class FeatureConstructionError(ValueError):
    """Raised when a causal feature cannot be computed safely."""


@dataclass(frozen=True, slots=True)
class RegressionSample:
    """Features known at an origin and its next-session delta target."""

    origin_date: date
    target_date: date
    origin_close: float
    actual_close: float
    target_delta: float
    feature_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegressionDataset:
    """Causal feature samples plus dated returns used for fold-local PACF."""

    feature_names: tuple[str, ...]
    samples: tuple[RegressionSample, ...]
    return_dates: tuple[date, ...]
    daily_returns: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise FeatureConstructionError("Regression dataset contains no usable samples")
        if len(self.return_dates) != len(self.daily_returns):
            raise FeatureConstructionError("Return dates and values must align")
        expected_width = len(self.feature_names)
        if any(len(sample.feature_values) != expected_width for sample in self.samples):
            raise FeatureConstructionError("Feature rows do not match feature_names")

    @property
    def target_dates(self) -> tuple[date, ...]:
        return tuple(sample.target_date for sample in self.samples)

    def matrix(
        self,
        samples: Sequence[RegressionSample] | None = None,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> FloatArray:
        chosen_samples = self.samples if samples is None else samples
        selected_names = self.feature_names if feature_names is None else tuple(feature_names)
        index_by_name = {name: index for index, name in enumerate(self.feature_names)}
        try:
            indices = [index_by_name[name] for name in selected_names]
        except KeyError as exc:
            raise FeatureConstructionError(f"Unknown feature requested: {exc.args[0]}") from exc
        return np.asarray(
            [[sample.feature_values[index] for index in indices] for sample in chosen_samples],
            dtype=np.float64,
        )

    @staticmethod
    def targets(samples: Sequence[RegressionSample]) -> FloatArray:
        return np.asarray([sample.target_delta for sample in samples], dtype=np.float64)

    @staticmethod
    def origin_closes(samples: Sequence[RegressionSample]) -> FloatArray:
        return np.asarray([sample.origin_close for sample in samples], dtype=np.float64)

    def returns_through(self, cutoff: date) -> FloatArray:
        """Return only daily returns observable by a fold's training cutoff."""

        return np.asarray(
            [value for value_date, value in zip(self.return_dates, self.daily_returns) if value_date <= cutoff],
            dtype=np.float64,
        )

    def samples_for_target_dates(
        self, target_dates: Iterable[date]
    ) -> tuple[RegressionSample, ...]:
        """Select exact common-plan targets without reordering or silent omission."""

        requested = tuple(target_dates)
        by_date = {sample.target_date: sample for sample in self.samples}
        missing = [value for value in requested if value not in by_date]
        if missing:
            formatted = ", ".join(value.isoformat() for value in missing[:5])
            raise FeatureConstructionError(f"Missing planned target dates: {formatted}")
        return tuple(by_date[value] for value in requested)


def _ema(values: Sequence[float], period: int) -> list[float]:
    alpha = 2.0 / (period + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=0))


def _safe_relative(numerator: float, denominator: float, *, name: str, origin: date) -> float:
    if denominator == 0.0:
        raise FeatureConstructionError(f"{name} has a zero denominator at {origin.isoformat()}")
    return numerator / denominator


def _minimum_origin_index(config: RegressionFeatureConfig) -> int:
    candidates = [
        max(config.return_lags),
        max(config.rolling_return_windows),
        max(config.volume_windows) - 1,
        config.rsi_period,
        config.ema_slow_period + config.macd_signal_period - 2,
        config.bollinger_window - 1,
        1,
    ]
    if config.raw_price_lags:
        candidates.append(max(config.raw_price_lags))
    return max(candidates)


def regression_feature_names(config: RegressionFeatureConfig) -> tuple[str, ...]:
    """Return the deterministic candidate-feature order."""

    names: list[str] = []
    names.extend(f"return_lag_{lag}" for lag in config.return_lags)
    for window in config.rolling_return_windows:
        names.extend((f"return_mean_{window}", f"return_std_{window}"))
    names.extend(("volume_log", "volume_change_1"))
    for window in config.volume_windows:
        names.extend((f"volume_ratio_{window}", f"volume_z_{window}"))
    names.extend(("range_pct", "open_close_spread_pct", "close_location_in_range"))
    names.extend(f"close_relative_sma_{window}" for window in config.rolling_return_windows)
    names.append(f"rsi_{config.rsi_period}")
    names.extend(
        (
            f"ema_relative_{config.ema_fast_period}",
            f"ema_relative_{config.ema_slow_period}",
            "macd_relative",
            "macd_signal_relative",
            "macd_histogram_relative",
            f"bollinger_z_{config.bollinger_window}",
            f"bollinger_width_{config.bollinger_window}",
            f"bollinger_position_{config.bollinger_window}",
        )
    )
    names.extend(f"raw_close_lag_{lag}" for lag in config.raw_price_lags)
    return tuple(names)


def build_regression_dataset(
    records: Sequence[OhlcvRecord],
    config: RegressionFeatureConfig = RegressionFeatureConfig(),
) -> RegressionDataset:
    """Build strictly causal predictors and Date[t+1] delta targets.

    Indicator warm-up rows are deliberately excluded. No raw value or derived
    feature is forward-filled or backward-filled.
    """

    require_chronological_records(records)
    minimum_origin = _minimum_origin_index(config)
    if len(records) <= minimum_origin + 1:
        raise FeatureConstructionError(
            f"Need more than {minimum_origin + 1} records for configured feature warm-up"
        )

    closes = [record.close for record in records]
    volumes = [record.volume for record in records]
    returns = [math.nan] + [
        _safe_relative(closes[index], closes[index - 1], name="daily return", origin=records[index].trading_date) - 1.0
        for index in range(1, len(records))
    ]
    ema_fast = _ema(closes, config.ema_fast_period)
    ema_slow = _ema(closes, config.ema_slow_period)
    macd = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
    macd_signal = _ema(macd, config.macd_signal_period)
    names = regression_feature_names(config)

    samples: list[RegressionSample] = []
    for index in range(minimum_origin, len(records) - 1):
        origin = records[index]
        target = records[index + 1]
        values: list[float] = []

        values.extend(returns[index - lag + 1] for lag in config.return_lags)
        for window in config.rolling_return_windows:
            return_window = returns[index - window + 1 : index + 1]
            mean_return, std_return = _mean_std(return_window)
            values.extend((mean_return, std_return))

        values.append(math.log1p(origin.volume))
        values.append(
            _safe_relative(
                origin.volume,
                records[index - 1].volume,
                name="volume change",
                origin=origin.trading_date,
            )
            - 1.0
        )
        for window in config.volume_windows:
            volume_window = volumes[index - window + 1 : index + 1]
            mean_volume, std_volume = _mean_std(volume_window)
            values.append(
                _safe_relative(
                    origin.volume,
                    mean_volume,
                    name=f"volume ratio {window}",
                    origin=origin.trading_date,
                )
                - 1.0
            )
            values.append(0.0 if std_volume == 0.0 else (origin.volume - mean_volume) / std_volume)

        values.append((origin.high - origin.low) / origin.close)
        values.append((origin.close - origin.open) / origin.open)
        day_range = origin.high - origin.low
        values.append(0.5 if day_range == 0.0 else (origin.close - origin.low) / day_range)
        for window in config.rolling_return_windows:
            mean_close, _ = _mean_std(closes[index - window + 1 : index + 1])
            values.append(origin.close / mean_close - 1.0)

        rsi_returns = returns[index - config.rsi_period + 1 : index + 1]
        average_gain = sum(max(value, 0.0) for value in rsi_returns) / config.rsi_period
        average_loss = sum(max(-value, 0.0) for value in rsi_returns) / config.rsi_period
        if average_gain == 0.0 and average_loss == 0.0:
            rsi = 50.0
        elif average_loss == 0.0:
            rsi = 100.0
        else:
            relative_strength = average_gain / average_loss
            rsi = 100.0 - (100.0 / (1.0 + relative_strength))
        values.append(rsi)

        values.extend(
            (
                origin.close / ema_fast[index] - 1.0,
                origin.close / ema_slow[index] - 1.0,
                macd[index] / origin.close,
                macd_signal[index] / origin.close,
                (macd[index] - macd_signal[index]) / origin.close,
            )
        )

        bollinger_closes = closes[index - config.bollinger_window + 1 : index + 1]
        bollinger_mean, bollinger_std = _mean_std(bollinger_closes)
        if bollinger_std == 0.0:
            bollinger_z = 0.0
            bollinger_width = 0.0
            bollinger_position = 0.5
        else:
            band_distance = config.bollinger_standard_deviations * bollinger_std
            lower_band = bollinger_mean - band_distance
            upper_band = bollinger_mean + band_distance
            bollinger_z = (origin.close - bollinger_mean) / bollinger_std
            bollinger_width = (upper_band - lower_band) / bollinger_mean
            bollinger_position = (origin.close - lower_band) / (upper_band - lower_band)
        values.extend((bollinger_z, bollinger_width, bollinger_position))

        values.extend(closes[index - lag] for lag in config.raw_price_lags)
        if not all(math.isfinite(value) for value in values):
            raise FeatureConstructionError(
                f"Non-finite derived feature at origin {origin.trading_date.isoformat()}"
            )
        samples.append(
            RegressionSample(
                origin_date=origin.trading_date,
                target_date=target.trading_date,
                origin_close=origin.close,
                actual_close=target.close,
                target_delta=target.close - origin.close,
                feature_values=tuple(values),
            )
        )

    dataset = RegressionDataset(
        feature_names=names,
        samples=tuple(samples),
        return_dates=tuple(record.trading_date for record in records[1:]),
        daily_returns=tuple(returns[1:]),
    )
    LOGGER.info(
        "Built causal LIR features samples=%d features=%d first_origin=%s last_origin=%s",
        len(dataset.samples),
        len(dataset.feature_names),
        dataset.samples[0].origin_date,
        dataset.samples[-1].origin_date,
    )
    return dataset


def feature_names_for_pacf_lags(
    candidate_names: Sequence[str], selected_lags: Sequence[int]
) -> tuple[str, ...]:
    """Keep non-return-lag features plus fold-selected return lags."""

    selected = {f"return_lag_{lag}" for lag in selected_lags}
    return tuple(
        name
        for name in candidate_names
        if not name.startswith("return_lag_") or name in selected
    )
