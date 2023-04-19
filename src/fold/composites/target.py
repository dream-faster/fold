# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey)<info@dreamfaster.ai> See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from ..base import Composite, InvertibleTransformation, Pipelines, T
from ..utils.checks import get_prediction_column, get_prediction_column_name
from ..utils.list import wrap_in_double_list_if_needed
from .common import get_concatenated_names


class TransformTarget(Composite):
    """
    Transforms the target within the context of the wrapped Pipeline.
    `wrapped_pipeline` will be applied to the input data, where the target (`y`) is already transformed.
    `y_pipeline` will be applied to the target column.

    The inverse of `y_transformation` will be applied to the predictions of the primary pipeline.

    Eg.: Log or Difference transformation.

    Parameters
    ----------
    wrapped_pipeline: Pipeline
        Pipeline, which will be applied to the input data, where the target (`y`) is already transformed.
    y_pipeline: Union[List[InvertibleTransformation], InvertibleTransformation]
        InvertibleTransformations, which will be applied to the target (`y`)

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.composites import ModelResiduals
    >>> from sklearn.linear_model import LinearRegression
    >>> from fold.transformations import Difference
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
    >>> pipeline = TransformTarget(
    ...     wrapped_pipeline=LinearRegression(),
    ...     y_pipeline=Difference(),
    ... )
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)

    ```
    """

    properties = Composite.Properties(
        primary_only_single_pipeline=True,
        secondary_only_single_pipeline=True,
    )

    def __init__(
        self,
        wrapped_pipeline: Pipelines,
        y_pipeline: Union[List[InvertibleTransformation], InvertibleTransformation],
    ) -> None:
        self.wrapped_pipeline = wrap_in_double_list_if_needed(wrapped_pipeline)
        self.y_pipeline = wrap_in_double_list_if_needed(y_pipeline)
        self.name = "TransformTarget-" + get_concatenated_names(
            self.wrapped_pipeline + self.y_pipeline
        )

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        # TransformTarget's primary transformation transforms `y`, not `X`.
        if y is None:
            return (
                pd.DataFrame(),
                None,
            )  # at inference time, `y` will be None, and we don't need to use primary transformations at all, so we return a dummy DataFrame.
        else:
            return y.to_frame(), None

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        raise NotImplementedError

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        results_primary: List[pd.DataFrame],
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T]:
        return X, results_primary[0].squeeze()

    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
    ) -> pd.DataFrame:
        predictions = get_prediction_column(secondary_results[0])
        for transformation in reversed(self.y_pipeline[0]):
            predictions = transformation.inverse_transform(predictions)
        orignal_results = secondary_results[0]
        orignal_results[
            get_prediction_column_name(orignal_results)
        ] = predictions.squeeze()
        return orignal_results

    def get_child_transformations_primary(self) -> Pipelines:
        return self.y_pipeline

    def get_child_transformations_secondary(
        self,
    ) -> Optional[Pipelines]:
        return self.wrapped_pipeline

    def clone(self, clone_child_transformations: Callable) -> TransformTarget:
        return TransformTarget(
            wrapped_pipeline=clone_child_transformations(self.wrapped_pipeline),
            y_pipeline=clone_child_transformations(self.y_pipeline),
        )
