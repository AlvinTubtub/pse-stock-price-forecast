"""Unit tests for target alignment and the common chronological split."""

from datetime import date, timedelta

import pytest

from config.model_config import ModelConfig, ModelId
from src.data.split import build_company_evaluation_plan, split_next_day_pairs
from src.data.validator import OhlcvRecord
from src.features.targets import build_next_day_pairs


def records(count: int) -> tuple[OhlcvRecord, ...]:
    start = date(2024, 1, 2)
    return tuple(
        OhlcvRecord(
            trading_date=start + timedelta(days=index),
            open=100.0 + index,
            high=102.0 + index,
            low=99.0 + index,
            close=101.0 + index,
            volume=1_000.0 + index,
        )
        for index in range(count)
    )


def test_next_day_target_alignment() -> None:
    source = records(3)
    pairs = build_next_day_pairs(source)

    assert len(pairs) == 2
    assert pairs[0].origin_date == source[0].trading_date
    assert pairs[0].target_date == source[1].trading_date
    assert pairs[0].origin_close == source[0].close
    assert pairs[0].actual_close == source[1].close
    assert pairs[0].target_delta == source[1].close - source[0].close
    assert pairs[1].origin_date == pairs[0].target_date


def test_configurable_development_evaluation_split() -> None:
    pairs = build_next_day_pairs(records(9))
    plan = split_next_day_pairs("ALI", pairs, evaluation_proportion=0.25)

    assert len(pairs) == 8
    assert len(plan.development_pairs) == 6
    assert len(plan.evaluation_pairs) == 2
    assert plan.evaluation_proportion == 0.25


def test_default_split_is_proportional_not_a_fixed_row_count() -> None:
    small = build_company_evaluation_plan("ALI", records(21))
    large = build_company_evaluation_plan("ALI", records(101))

    assert len(small.evaluation_pairs) == 3
    assert len(large.evaluation_pairs) == 15
    assert len(small.evaluation_pairs) != 243
    assert len(large.evaluation_pairs) != 243


def test_development_and_evaluation_targets_do_not_overlap() -> None:
    plan = build_company_evaluation_plan(
        "ALI", records(20), model_config=ModelConfig(evaluation_proportion=0.2)
    )

    assert set(plan.development_target_dates).isdisjoint(plan.evaluation_target_dates)


def test_split_has_no_future_leakage() -> None:
    plan = build_company_evaluation_plan("ALI", records(20))

    assert max(plan.development_target_dates) < min(plan.evaluation_target_dates)
    assert all(
        pair.origin_date < pair.target_date
        for pair in plan.development_pairs + plan.evaluation_pairs
    )


def test_all_future_models_share_one_common_target_date_plan() -> None:
    plan = build_company_evaluation_plan("ALI", records(30))
    dates_given_to_each_model = {
        model_id: plan.evaluation_target_dates
        for model_id in (
            ModelId.LAG_REGRESSION,
            ModelId.ARIMA,
            ModelId.LSTM,
            ModelId.NAIVE,
        )
    }

    assert len(set(dates_given_to_each_model.values())) == 1
    assert next(iter(dates_given_to_each_model.values())) == tuple(
        pair.target_date for pair in plan.evaluation_pairs
    )


@pytest.mark.parametrize("proportion", [0.0, 1.0, -0.1, 1.1])
def test_invalid_evaluation_proportion_is_rejected(proportion: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        split_next_day_pairs("ALI", build_next_day_pairs(records(4)), evaluation_proportion=proportion)
