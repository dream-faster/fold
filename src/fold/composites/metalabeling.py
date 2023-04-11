from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from fold.composites.common import get_concatenated_names
from fold.utils.checks import get_prediction_column
from fold.utils.list import wrap_in_double_list_if_needed

from ..base import BlocksOrWrappable, Composite, Pipelines, T


class MetaLabeling(Composite):
    """
    MetaLabeling takes a primary pipeline and a meta pipeline.
    The primary pipeline is used to predict the target variable.
    The meta pipeline is used to predict whether the primary model's prediction's are correct.
    It multiplies the probabilities from the meta pipeline with the predictions of the primary pipeline.

    It's only applicable for binary classification problems, where the labels are either `1`, `-1` or one of them are zero.

    Parameters
    ----------

    primary : BlocksOrWrappable
        A pipeline to be applied to the data.

    meta : BlocksOrWrappable
        A pipeline to be applied to the data.

    positive_class : Union[int, float]
        The positive class of the primary pipeline.

    primary_output_included :  bool, optional
        Whether the primary pipeline's output is included in the meta pipeline's input, by default False.

    Outputs
    -------
        A prediction is a float between 0 and 1.
        It does not output probabilities, as the prediction already includes the probabilities.

    """

    properties = Composite.Properties(
        primary_requires_predictions=True,
        primary_only_single_pipeline=True,
        secondary_requires_predictions=True,
        secondary_only_single_pipeline=True,
    )

    def __init__(
        self,
        primary: BlocksOrWrappable,
        meta: BlocksOrWrappable,
        positive_class: Union[int, float],
        primary_output_included: bool = False,
    ) -> None:
        self.primary = wrap_in_double_list_if_needed(primary)
        self.meta = wrap_in_double_list_if_needed(meta)
        self.positive_class = positive_class
        self.primary_output_included = primary_output_included
        self.name = "MetaLabeling-" + get_concatenated_names(self.primary + self.meta)

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T]:
        X = (
            pd.concat([X] + results_primary, axis="columns")
            if self.primary_output_included
            else X
        )
        predictions = get_prediction_column(results_primary[0])
        y = y.astype(int) == predictions.astype(int)
        return X, y

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        raise NotImplementedError

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
    ) -> pd.DataFrame:
        primary_predictions = get_prediction_column(primary_results[0])
        meta_probabilities = secondary_results[0][
            [
                col
                for col in secondary_results[0].columns
                if col.startswith("probabilities_")
            ]
        ]
        meta_probabilities_positive_class = meta_probabilities[
            [
                col
                for col in meta_probabilities.columns
                if col.split("_")[-1] == str(self.positive_class)
            ]
        ]
        if len(meta_probabilities_positive_class.columns) != 1:
            raise ValueError(
                "Meta pipeline needs to be concluded with probabilities of the"
                f" positive class: {str(self.positive_class)}"
            )
        result = (
            primary_predictions * meta_probabilities_positive_class.squeeze()
        ).rename(f"predictions_{self.name}")
        dc = {
            col: f"probabilities_{self.name}_" + col.split("_")[-1]
            for col in meta_probabilities.columns
        }
        meta_probabilities = meta_probabilities.rename(columns=dc)
        return pd.concat([result, meta_probabilities], axis="columns")

    def get_child_transformations_primary(self) -> Pipelines:
        return self.primary

    def get_child_transformations_secondary(
        self,
    ) -> Optional[Pipelines]:
        return self.meta

    def clone(self, clone_child_transformations: Callable) -> MetaLabeling:
        return MetaLabeling(
            primary=clone_child_transformations(self.primary),
            meta=clone_child_transformations(self.meta),
            positive_class=self.positive_class,
            primary_output_included=self.primary_output_included,
        )
