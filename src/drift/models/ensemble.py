from copy import deepcopy
from typing import Callable, List

import pandas as pd

from ..transformations.base import Composite, Transformations
from ..utils.list import unique, wrap_in_list


class Ensemble(Composite):
    def __init__(self, models: Transformations) -> None:
        self.models = models
        self.name = "Ensemble-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in models
            ]
        )

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        results_have_probabilities = all(
            [
                any([True for col in df.columns if col.startswith("probabilities_")])
                for df in results
            ]
        )
        if results_have_probabilities:
            return get_groupped_columns_classification(results, self.name)
        else:
            return get_groupped_columns_regression(results, self.name)

    def get_child_transformations(self) -> Transformations:
        return self.models

    def clone(self, clone_child_transformations: Callable) -> Composite:
        return Ensemble(
            models=clone_child_transformations(self.models),
        )


class PerColumnEnsemble(Composite):
    def __init__(self, models: Transformations) -> None:
        self.models = wrap_in_list(models)
        self.name = "PerColumnEnsemble-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in self.models
            ]
        )

    def before_fit(self, X: pd.DataFrame) -> None:
        self.models = [deepcopy(self.models) for _ in X.columns]

    def preprocess_X(
        self, X: pd.DataFrame, index: int, for_inference: bool
    ) -> pd.DataFrame:
        return X.iloc[:, index].to_frame()

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        results_have_probabilities = all(
            [
                any([True for col in df.columns if col.startswith("probabilities_")])
                for df in results
            ]
        )
        if results_have_probabilities:
            return get_groupped_columns_classification(results, self.name)
        else:
            return get_groupped_columns_regression(results, self.name)

    def get_child_transformations(self) -> Transformations:
        return self.models

    def clone(self, clone_child_transformations: Callable) -> Composite:
        return PerColumnEnsemble(
            models=clone_child_transformations(self.models),
        )


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
            axis=1,
        )
        .mean(axis=1)
        .rename(f"predictions_Ensemble_{name}")
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
            axis=1,
        )
        .mean(axis=1)
        .rename(f"predictions_{name}")
    )

    probabilities = [
        (
            pd.concat(
                [
                    df[[col for col in df.columns if col.endswith(selected_class)]].squeeze()
                    for df in results
                ],
                axis=1,
            )
            .mean(axis=1)
            .rename(f"probabilities_{name}_{selected_class}")
        )
        for selected_class in classes
    ]
    return pd.concat([predictions] + probabilities, axis=1)
