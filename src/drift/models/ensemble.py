from __future__ import annotations

from copy import deepcopy
from typing import Callable, List, Optional

import pandas as pd

from ..transformations.base import Composite, Transformations, TransformationsAlwaysList
from ..transformations.common import get_concatenated_names
from ..utils.checks import all_have_probabilities
from ..utils.list import unique, wrap_in_list


class Ensemble(Composite):

    properties = Composite.Properties()

    def __init__(self, models: Transformations) -> None:
        self.models = models
        self.name = "Ensemble-" + get_concatenated_names(models)

    def postprocess_result_primary(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return postprocecess_results(results, self.name)

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.models

    def clone(self, clone_child_transformations: Callable) -> Ensemble:
        return Ensemble(
            models=clone_child_transformations(self.models),
        )


class PerColumnEnsemble(Composite):

    properties = Composite.Properties()

    def __init__(self, models: Transformations) -> None:
        self.models: TransformationsAlwaysList = wrap_in_list(models)
        self.name = "PerColumnEnsemble-" + get_concatenated_names(self.models)

    def before_fit(self, X: pd.DataFrame) -> None:
        self.models = [deepcopy(self.models) for _ in X.columns]

    def preprocess_X_primary(
        self, X: pd.DataFrame, index: int, y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return X.iloc[:, index].to_frame()

    def postprocess_result_primary(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return postprocecess_results(results, self.name)

    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.models

    def clone(self, clone_child_transformations: Callable) -> PerColumnEnsemble:
        return PerColumnEnsemble(
            models=clone_child_transformations(self.models),
        )


def postprocecess_results(
    results: List[pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    if all_have_probabilities(results):
        return get_groupped_columns_classification(results, name)
    else:
        return get_groupped_columns_regression(results, name)


def get_groupped_columns_regression(
    results: List[pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    return (
        pd.concat(
            [
                df[
                    [col for col in df.columns if col.startswith("predictions_")]
                ].squeeze()
                for df in results
            ],
            axis="columns",
        )
        .mean(axis="columns")
        .rename(f"predictions_{name}")
        .to_frame()
    )


def get_groupped_columns_classification(
    results: List[pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    columns = results[0].columns.to_list()
    probabilities_columns = [col for col in columns if col.startswith("probabilities_")]
    classes = unique([line.split("_")[-1] for line in probabilities_columns])

    predictions = (
        pd.concat(
            [
                df[
                    [col for col in df.columns if col.startswith("predictions_")]
                ].squeeze()
                for df in results
            ],
            axis="columns",
        )
        .mean(axis="columns")
        .rename(f"predictions_{name}")
    )

    probabilities = [
        (
            pd.concat(
                [
                    df[
                        [
                            col
                            for col in df.columns
                            if col.startswith("probabilities_")
                            and col.split("_")[-1] == selected_class
                        ]
                    ].squeeze()
                    for df in results
                ],
                axis="columns",
            )
            .mean(axis="columns")
            .rename(f"probabilities_{name}_{selected_class}")
        )
        for selected_class in classes
    ]
    return pd.concat([predictions] + probabilities, axis="columns")
