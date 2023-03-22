from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd

from fold.composites.common import get_concatenated_names
from fold.utils.checks import get_prediction_column
from fold.utils.list import wrap_in_double_list_if_needed

from ..transformations.base import BlocksOrWrappable, Pipelines
from .base import Composite, T


class Hybrid(Composite):

    """
    This is a composite transformation that takes a primary pipeline and a meta pipeline.
    The primary pipeline is used to predict the target variable.
    The meta pipeline is used to predict the primary pipeline's residual.
    It adds together the primary pipeline's output with the predicted residual.

    Also known as:
    - Residual chasing
    - ?

    It's only applicable for regression tasks.
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
        primary_output_included: bool = False,
    ) -> None:
        self.primary = wrap_in_double_list_if_needed(primary)
        self.meta = wrap_in_double_list_if_needed(meta)
        self.primary_output_included = primary_output_included
        self.name = "Hybrid-" + get_concatenated_names(self.primary + self.meta)

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        return X, y

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
        residuals = y - predictions
        return X, residuals

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
        residual_predictions = get_prediction_column(secondary_results[0])

        return (
            (primary_predictions + residual_predictions)
            .rename(f"predictions_{self.name}")
            .to_frame()
        )

    def get_child_transformations_primary(self) -> Pipelines:
        return self.primary

    def get_child_transformations_secondary(
        self,
    ) -> Optional[Pipelines]:
        return self.meta

    def clone(self, clone_child_transformations: Callable) -> Hybrid:
        return Hybrid(
            primary=clone_child_transformations(self.primary),
            meta=clone_child_transformations(self.meta),
            primary_output_included=self.primary_output_included,
        )
