"""Canonical dated OOS prediction records and strict cross-model alignment."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math

from config.model_config import ModelId
from src.data.split import CompanyEvaluationPlan


REQUIRED_EVALUATION_MODELS: tuple[ModelId, ...] = (
    ModelId.LAG_REGRESSION,
    ModelId.ARIMA,
    ModelId.LSTM,
    ModelId.NAIVE,
)


class BacktestAlignmentError(ValueError):
    """Raised when canonical model/date combinations are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class CanonicalPrediction:
    """One genuine one-step-ahead Close prediction at a declared target date."""

    symbol: str
    model: ModelId
    origin_date: date
    target_date: date
    origin_close: float
    actual_close: float
    predicted_close: float
    error: float

    def __post_init__(self) -> None:
        if not self.symbol or self.symbol != self.symbol.upper():
            raise BacktestAlignmentError("Prediction symbol must be non-empty and uppercase")
        if self.origin_date >= self.target_date:
            raise BacktestAlignmentError("Prediction origin must precede its target date")
        numeric = (
            self.origin_close,
            self.actual_close,
            self.predicted_close,
            self.error,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise BacktestAlignmentError("Prediction prices and error must be finite")
        expected_error = self.predicted_close - self.actual_close
        if self.error != expected_error:
            raise BacktestAlignmentError("Prediction error must equal predicted minus actual")

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        model: ModelId,
        origin_date: date,
        target_date: date,
        origin_close: float,
        actual_close: float,
        predicted_close: float,
    ) -> "CanonicalPrediction":
        predicted = float(predicted_close)
        actual = float(actual_close)
        return cls(
            symbol=symbol,
            model=model,
            origin_date=origin_date,
            target_date=target_date,
            origin_close=float(origin_close),
            actual_close=actual,
            predicted_close=predicted,
            error=predicted - actual,
        )

    def as_dict(self) -> dict[str, str | float]:
        return {
            "symbol": self.symbol,
            "model": self.model.value,
            "origin_date": self.origin_date.isoformat(),
            "target_date": self.target_date.isoformat(),
            "origin_close": self.origin_close,
            "actual_close": self.actual_close,
            "predicted_close": self.predicted_close,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BacktestData:
    """Strictly aligned OOS records; future frontend arrays must derive from this."""

    symbol: str
    target_dates: tuple[date, ...]
    actual_closes: tuple[float, ...]
    records_by_model: tuple[tuple[ModelId, tuple[CanonicalPrediction, ...]], ...]

    def records_for(self, model: ModelId) -> tuple[CanonicalPrediction, ...]:
        try:
            return dict(self.records_by_model)[model]
        except KeyError as exc:
            raise BacktestAlignmentError(f"No backtest records for model {model.value}") from exc

    @property
    def all_records(self) -> tuple[CanonicalPrediction, ...]:
        return tuple(
            record
            for _, model_records in self.records_by_model
            for record in model_records
        )

    def predicted_closes(self, model: ModelId) -> tuple[float, ...]:
        return tuple(record.predicted_close for record in self.records_for(model))

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "target_dates": [value.isoformat() for value in self.target_dates],
            "actual_closes": list(self.actual_closes),
            "predicted_closes_by_model": {
                model.value: [record.predicted_close for record in records]
                for model, records in self.records_by_model
            },
            "records": [record.as_dict() for record in self.all_records],
            "source": "chronological_out_of_sample_evaluation",
        }


def canonical_records_from_arrays(
    plan: CompanyEvaluationPlan,
    *,
    model: ModelId,
    target_dates: Sequence[date],
    actual_closes: Sequence[float],
    predicted_closes: Sequence[float],
) -> tuple[CanonicalPrediction, ...]:
    """Join one model's dated output to canonical evaluation origins and actuals."""

    dates = tuple(target_dates)
    actuals = tuple(float(value) for value in actual_closes)
    predictions = tuple(float(value) for value in predicted_closes)
    if not (len(dates) == len(actuals) == len(predictions)):
        raise BacktestAlignmentError(
            f"Model {model.value} dates, actuals, and predictions must have equal length"
        )
    if len(set(dates)) != len(dates):
        raise BacktestAlignmentError(f"Model {model.value} contains duplicate target dates")
    supplied = {
        target_date: (actual, prediction)
        for target_date, actual, prediction in zip(dates, actuals, predictions, strict=True)
    }
    expected_dates = plan.evaluation_target_dates
    if set(supplied) != set(expected_dates):
        missing = sorted(set(expected_dates) - set(supplied))
        extra = sorted(set(supplied) - set(expected_dates))
        raise BacktestAlignmentError(
            f"Model {model.value} target-date mismatch; "
            f"missing={[value.isoformat() for value in missing]} "
            f"extra={[value.isoformat() for value in extra]}"
        )
    records: list[CanonicalPrediction] = []
    for pair in plan.evaluation_pairs:
        supplied_actual, predicted = supplied[pair.target_date]
        if supplied_actual != pair.actual_close:
            raise BacktestAlignmentError(
                f"Model {model.value} actual Close disagrees on {pair.target_date}"
            )
        records.append(
            CanonicalPrediction.create(
                symbol=plan.symbol,
                model=model,
                origin_date=pair.origin_date,
                target_date=pair.target_date,
                origin_close=pair.origin_close,
                actual_close=pair.actual_close,
                predicted_close=predicted,
            )
        )
    return tuple(records)


def naive_prediction_records(
    plan: CompanyEvaluationPlan,
) -> tuple[CanonicalPrediction, ...]:
    """Create the date-indexed one-step persistence benchmark Close[t+1]=Close[t]."""

    return tuple(
        CanonicalPrediction.create(
            symbol=plan.symbol,
            model=ModelId.NAIVE,
            origin_date=pair.origin_date,
            target_date=pair.target_date,
            origin_close=pair.origin_close,
            actual_close=pair.actual_close,
            predicted_close=pair.origin_close,
        )
        for pair in plan.evaluation_pairs
    )


def align_backtest_records(
    plan: CompanyEvaluationPlan,
    records_by_model: Mapping[ModelId, Sequence[CanonicalPrediction]],
) -> BacktestData:
    """Require every model/date combination and align records in plan date order."""

    supplied_models = set(records_by_model)
    required_models = set(REQUIRED_EVALUATION_MODELS)
    if supplied_models != required_models:
        missing = sorted(model.value for model in required_models - supplied_models)
        extra = sorted(model.value for model in supplied_models - required_models)
        raise BacktestAlignmentError(
            f"Model-set mismatch; missing={missing} extra={extra}"
        )
    expected_dates = plan.evaluation_target_dates
    expected_by_date = {pair.target_date: pair for pair in plan.evaluation_pairs}
    aligned: list[tuple[ModelId, tuple[CanonicalPrediction, ...]]] = []
    for model in REQUIRED_EVALUATION_MODELS:
        records = tuple(records_by_model[model])
        by_date = {record.target_date: record for record in records}
        if len(by_date) != len(records):
            raise BacktestAlignmentError(f"Model {model.value} has duplicate target dates")
        if set(by_date) != set(expected_dates):
            missing = sorted(set(expected_dates) - set(by_date))
            extra = sorted(set(by_date) - set(expected_dates))
            raise BacktestAlignmentError(
                f"Model {model.value} target-date mismatch; "
                f"missing={[value.isoformat() for value in missing]} "
                f"extra={[value.isoformat() for value in extra]}"
            )
        model_records = tuple(by_date[value] for value in expected_dates)
        for record in model_records:
            pair = expected_by_date[record.target_date]
            if record.model is not model or record.symbol != plan.symbol:
                raise BacktestAlignmentError("Prediction model or symbol is inconsistent")
            if (
                record.origin_date != pair.origin_date
                or record.origin_close != pair.origin_close
                or record.actual_close != pair.actual_close
            ):
                raise BacktestAlignmentError(
                    f"Model {model.value} record disagrees with plan on {record.target_date}"
                )
            if record.target_date <= plan.development_target_dates[-1]:
                raise BacktestAlignmentError("In-sample prediction entered OOS backtest")
        aligned.append((model, model_records))
    return BacktestData(
        symbol=plan.symbol,
        target_dates=expected_dates,
        actual_closes=tuple(pair.actual_close for pair in plan.evaluation_pairs),
        records_by_model=tuple(aligned),
    )
