# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd

from fold.composites.common import get_concatenated_names
from fold.utils.checks import get_prediction_column
from fold.utils.list import wrap_in_double_list_if_needed

from ..base import Composite, Pipeline, Pipelines, T


class ModelResiduals(Composite):
    """
    This is a composite that combines two pipelines:
    * The primary pipeline is used to predict the target variable.
    * The meta pipeline is used to predict the primary pipeline's residual (or, error).

    It adds together the primary pipeline's output with the predicted residual.

    Also known as:
    - Residual chasing
    - Residual boosting
    - Hybrid approach
    - "Moving Average" in ARIMA

    It's only applicable for regression tasks.

    Parameters
    ----------

    primary : Pipeline
        A pipeline to be applied to the data. The target (`y`) is unchanged.
    meta : Pipeline
        A pipeline to predict the primary pipeline's residual. The target (`y`) is the primary pipeline's residual (or, error).
    primary_output_included : bool, optional
        Whether the primary pipeline's output is included in the meta pipeline's input, by default False.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.composites import ModelResiduals
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
        >>> pipeline = ModelResiduals(
        ...     primary=LinearRegression(),
        ...     meta=RandomForestRegressor(),
        ... )
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)

    References
    ----------
    - https://www.kaggle.com/code/ryanholbrook/hybrid-models
    - https://www.uber.com/en-DE/blog/m4-forecasting-competition/
    """

    properties = Composite.Properties(
        primary_requires_predictions=True,
        primary_only_single_pipeline=True,
        secondary_requires_predictions=True,
        secondary_only_single_pipeline=True,
    )

    def __init__(
        self,
        primary: Pipeline,
        meta: Pipeline,
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

    def clone(self, clone_child_transformations: Callable) -> ModelResiduals:
        return ModelResiduals(
            primary=clone_child_transformations(self.primary),
            meta=clone_child_transformations(self.meta),
            primary_output_included=self.primary_output_included,
        )
