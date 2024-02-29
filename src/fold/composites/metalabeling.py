# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from finml_utils.dataframes import concat_on_columns

from ..base import Artifact, Composite, Pipeline, Pipelines, T, get_concatenated_names
from ..utils.checks import (
    get_prediction_column,
    get_probabilities_columns,
)
from ..utils.list import wrap_in_double_list_if_needed


class MetaLabeling(Composite):
    """
    MetaLabeling takes a primary pipeline and a meta pipeline.
    The primary pipeline is used to predict the target variable.
    The meta pipeline is used to predict whether the primary model's prediction's are correct (a binary classification problem).
    It multiplies the probabilities from the meta pipeline with the predictions of the primary pipeline.

    It's only applicable for binary classification problems, where the labels are either `1`, `-1` or one of them are zero.

    Parameters
    ----------

    primary : Pipeline
        A pipeline to be applied to the data. Target (`y`) is unchanged.
    meta : Pipeline
        A pipeline to be applied to predict whether the primary pipeline's predictions are correct. Target (`y`) is `preds == y`.
    primary_output_included :  bool, optional
        Whether the primary pipeline's output is included in the meta pipeline's input, by default False.


    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SingleWindowSplitter
        >>> from fold.composites import MetaLabeling
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from fold.utils.tests import generate_zeros_and_ones
        >>> X, y  = generate_zeros_and_ones()
        >>> splitter = SingleWindowSplitter(train_window=0.5)
        >>> pipeline = MetaLabeling(
        ...     primary=LogisticRegression(),
        ...     meta=RandomForestClassifier(),
        ... )
        >>> preds, trained_pipeline, _, _ = train_backtest(pipeline, X, y, splitter)


    Outputs
    -------
        A prediction is a float between -1 or 0, and 1.
        It does not output probabilities, as the prediction already includes that information.


    References
    ----------
    [Meta Labeling (A Toy Example)](https://hudsonthames.org/meta-labeling-a-toy-example/)
    [Meta-Labeling: Theory and Framework](https://jfds.pm-research.com/content/4/3/31)

    """

    def __init__(
        self,
        primary: Pipeline,
        meta: Pipeline,
        primary_output_included: bool = False,
        name: str | None = None,
    ) -> None:
        self.primary = wrap_in_double_list_if_needed(primary)
        self.meta = wrap_in_double_list_if_needed(meta)
        self.primary_output_included = primary_output_included
        self.name = name or "MetaLabeling-" + get_concatenated_names(
            self.primary + self.meta
        )
        self.properties = Composite.Properties(
            primary_requires_predictions=True,
            primary_only_single_pipeline=True,
            secondary_requires_predictions=True,
            secondary_only_single_pipeline=True,
        )
        self.metadata = None

    def postprocess_result_primary(
        self,
        results: list[pd.DataFrame],
        y: pd.Series | None,
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return results[0]

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        artifact: Artifact,
        results_primary: pd.DataFrame,
        index: int,
        fit: bool,
    ) -> tuple[pd.DataFrame, T, Artifact]:
        X = (
            concat_on_columns([X, results_primary])
            if self.primary_output_included
            else X
        )
        if y is None:
            return X, y, artifact

        predictions = get_prediction_column(results_primary)
        y = (y.astype(int) == predictions.astype(int)).astype(float)
        events = Artifact.get_events(artifact)
        if events is None:
            return X, y, events
        events = events.copy()
        events["event_label"] = y
        return X, y, events

    def postprocess_result_secondary(
        self,
        primary_results: pd.DataFrame,
        secondary_results: list[pd.DataFrame],
        y: pd.Series | None,
        in_sample: bool,
    ) -> pd.DataFrame:
        primary_predictions = get_prediction_column(primary_results).mul(
            get_prediction_column(secondary_results[0])
        )
        return concat_on_columns(
            [primary_predictions, get_probabilities_columns(secondary_results[0])],
        )

    def postprocess_artifacts_secondary(
        self,
        primary_artifacts: pd.DataFrame,
        secondary_artifacts: list[Artifact],
        original_artifact: Artifact,
    ) -> pd.DataFrame:
        return primary_artifacts

    def get_children_primary(self, only_traversal: bool) -> Pipelines:
        return self.primary

    def get_children_secondary(
        self,
    ) -> Pipelines | None:
        return self.meta

    def clone(self, clone_children: Callable) -> MetaLabeling:
        clone = MetaLabeling(
            primary=clone_children(self.primary),
            meta=clone_children(self.meta),
            primary_output_included=self.primary_output_included,
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        clone.id = self.id
        return clone


def get_int_class(input: str) -> int:
    if input.endswith(".0"):
        return int(float(input[:-2]))
    if input == "True":
        return 1
    if input == "False":
        return 0

    return int(input)
