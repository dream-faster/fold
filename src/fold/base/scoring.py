from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pandas as pd
from finml_utils.dataframes import concat_on_columns

from ..utils.checks import (
    all_have_probabilities,
    get_prediction_column,
    get_probabilities_columns,
)
from .classes import Artifact, OutOfSamplePredictions

if TYPE_CHECKING and importlib.util.find_spec("krisi") is not None:
    from krisi import ScoreCard


def score_results(
    result: pd.DataFrame,
    y: pd.Series,
    artifacts: Artifact,
    krisi_args: dict | None = None,
) -> tuple[ScoreCard, OutOfSamplePredictions]:
    y, pred_point, probabilities, test_sample_weights = align_result_with_events(
        y=y,
        result=result,
        artifacts=artifacts,
    )
    from krisi import score

    return (
        score(
            y=y,
            predictions=pred_point,
            probabilities=probabilities,
            sample_weight=test_sample_weights,
            **(krisi_args if krisi_args is not None else {}),
        ),
        concat_on_columns([pred_point, probabilities]),
    )


def align_result_with_events(
    y: pd.Series,
    result: pd.DataFrame,
    artifacts: Artifact,
) -> tuple[pd.Series, pd.Series, pd.DataFrame | None, pd.Series | None]:
    events = Artifact.get_events(artifacts)
    test_sample_weights = Artifact.get_test_sample_weights(artifacts)

    probabilities = (
        get_probabilities_columns(result) if all_have_probabilities([result]) else None
    )
    pred_point = get_prediction_column(result)

    if events is not None:
        events = events.reindex(result.index).dropna()
        y = events.event_label
        test_sample_weights = events.event_test_sample_weights
        pred_point = pred_point.dropna()
        probabilities = probabilities.dropna() if probabilities is not None else None
    else:
        test_sample_weights = (
            test_sample_weights[pred_point.index]
            if test_sample_weights is not None
            else None
        )
        y = y[pred_point.index]

    return y, pred_point, probabilities, test_sample_weights
