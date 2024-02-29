# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from collections.abc import Callable

from finml_utils.dataframes import concat_on_columns
from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series as Series

from fold.base.classes import Artifact

from ..base import Composite, Pipeline, Pipelines, get_concatenated_names
from ..utils.checks import (
    get_prediction_column,
    get_probabilities_columns,
)
from ..utils.list import wrap_in_double_list_if_needed


class StrategyMetaLabeling(Composite):
    def __init__(
        self,
        meta: Pipeline,
        primary_output_included: bool = False,
        name: str | None = None,
    ) -> None:
        self.meta = wrap_in_double_list_if_needed(meta)
        self.primary_output_included = primary_output_included
        self.name = name or "SignalMetaLabeling-" + get_concatenated_names(self.meta)
        self.properties = Composite.Properties(
            primary_requires_predictions=True,
            primary_only_single_pipeline=True,
        )
        self.metadata = None

    def postprocess_result_primary(
        self,
        results: list[DataFrame],
        y: Series | None,
        original_artifact: Artifact,
        fit: bool,
    ) -> DataFrame:
        events = Artifact.get_events(original_artifact)
        if "event_strategy_label" not in events.columns:
            raise ValueError(
                "The events artifact does not contain the event_strategy_label column."
            )
        predictions = events.event_strategy_label.mul(
            get_prediction_column(results[0])
        ).rename(get_prediction_column(results[0]).name)
        return concat_on_columns([predictions, get_probabilities_columns(results[0])])

    def get_children_primary(self, only_traversal: bool) -> Pipelines:
        return self.meta

    def clone(self, clone_children: Callable) -> StrategyMetaLabeling:
        clone = StrategyMetaLabeling(
            meta=clone_children(self.meta),
            primary_output_included=self.primary_output_included,
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        clone.id = self.id
        return clone
