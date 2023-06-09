import importlib
from typing import Optional, Tuple

import pandas as pd

from ..utils.checks import (
    all_have_probabilities,
    get_prediction_column,
    get_probabilities_columns,
)
from ..utils.enums import ParsableEnum
from .classes import Artifact, Extras


class ScoreOn(ParsableEnum):
    predictions = "predictions"
    probabilities = "probabilities"


def score_results(
    result: pd.DataFrame,
    y: pd.Series,
    extras: Extras,
    artifacts: Artifact,
    sample_weights: Optional[pd.Series],
    krisi_args: Optional[dict] = None,
):
    y, pred_point, probabilities, test_sample_weights = align_result_with_events(
        y=y,
        sample_weights=sample_weights,
        result=result,
        extras=extras,
        artifacts=artifacts,
    )
    if importlib.util.find_spec("krisi") is not None:
        from krisi import score

        return score(
            y=y[pred_point.index],
            predictions=pred_point,
            probabilities=probabilities,
            sample_weight=test_sample_weights[pred_point.index]
            if sample_weights is not None
            else None,
            **(krisi_args if krisi_args is not None else {}),
        )
    else:
        raise ImportError(
            "krisi not installed. Please install krisi to use this function."
        )


def align_result_with_events(
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    result: pd.DataFrame,
    extras: Extras,
    artifacts: Artifact,
) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    labels = __get_labels(extras, artifacts)
    # TODO: override sample_weights if passed in

    probabilities = (
        get_probabilities_columns(result) if all_have_probabilities([result]) else None
    )
    pred_point = get_prediction_column(result)

    if labels is not None:
        y = labels.reindex(result.index).dropna()
        sample_weights = (
            __get_test_sample_weights(extras, artifacts).reindex(result.index).dropna()
        )
        pred_point = pred_point.dropna()
        probabilities = probabilities.dropna() if probabilities is not None else None
        sample_weights = sample_weights.dropna() if sample_weights is not None else None
    if len(y) != len(pred_point):
        pred_point = pred_point[: len(y)]
        probabilities = probabilities[: len(y)] if probabilities is not None else None
    sample_weights = sample_weights[y.index] if sample_weights is not None else None

    return y, pred_point, probabilities, sample_weights


def __get_labels(extras: Extras, artifacts: Artifact) -> Optional[pd.Series]:
    if artifacts is not None and "label" in artifacts.columns:
        return artifacts["label"]
    elif extras.events is not None:
        return extras.events["label"]
    else:
        return None


def __get_test_sample_weights(
    extras: Extras, artifacts: Artifact
) -> Optional[pd.Series]:
    if artifacts is not None and "sample_weights" in artifacts.columns:
        return artifacts["test_sample_weights"]
    elif extras.events is not None:
        return extras.events["test_sample_weights"]
    else:
        raise ValueError("No sample weights found in extras or artifacts")
