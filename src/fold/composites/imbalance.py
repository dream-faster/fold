# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ..base import Artifact, Composite, Pipelines, get_concatenated_names
from ..utils.checks import get_prediction_column_name, get_probabilities_columns
from ..utils.list import wrap_in_double_list_if_needed


def roc_youden_statistic(
    true_label: pd.Series, predicted_proba: pd.Series, pos_label: int
) -> float:
    fpr, tpr, thresholds = roc_curve(true_label, predicted_proba, pos_label=pos_label)

    thresholds = [threshold if threshold <= 1.0 else np.inf for threshold in thresholds]

    sorted_thresholds = sorted(
        list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True
    )

    optimal_threshold = [
        (ratio, threshold)
        for ratio, threshold in sorted_thresholds
        if threshold != np.inf
    ]

    return optimal_threshold[0][1]


class FindThreshold(Composite):
    pos_label: int

    def __init__(
        self, pipelines: Pipelines, name: Optional[str] = None, pos_label: int = 1
    ) -> None:
        self.pipelines = wrap_in_double_list_if_needed(pipelines)

        self.name = name or "FindThreshold-" + get_concatenated_names(pipelines)
        self.properties = Composite.Properties()
        self.threshold = None
        self.pos_label = pos_label

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        model_result = results[0]
        pos_results = get_probabilities_columns(model_result).iloc[:, self.pos_label]
        if self.threshold is None and y is not None:
            self.threshold = roc_youden_statistic(y, pos_results, self.pos_label)

        res = pd.Series(
            [abs(1 - self.pos_label)] * len(model_result),
            index=model_result.index,
        )
        res.loc[pos_results >= self.threshold] = self.pos_label

        model_result[get_prediction_column_name(model_result)] = res

        return model_result

    def postprocess_artifacts_primary(
        self,
        primary_artifacts: List[Artifact],
        results: List[pd.DataFrame],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return pd.concat(
            primary_artifacts
            + [
                pd.Series(
                    [self.threshold],
                    index=primary_artifacts[0].index[:1],
                    name="threshold",
                ),
            ],
            axis="columns",
        )

    def get_children_primary(self) -> Pipelines:
        return self.pipelines

    def clone(self, clone_children: Callable) -> FindThreshold:
        clone = FindThreshold(
            pipelines=clone_children(self.pipelines), pos_label=self.pos_label
        )
        clone.properties = self.properties
        clone.name = self.name
        return clone
