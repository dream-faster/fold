# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from fold.base.classes import Artifact

from ..base import (
    Composite,
    InvertibleTransformation,
    Pipeline,
    Pipelines,
    T,
    get_concatenated_names,
)
from ..utils.checks import get_prediction_column, get_prediction_column_name
from ..utils.dataframe import to_series
from ..utils.list import wrap_in_double_list_if_needed


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
    invert_wrapped_output: bool, default=True
        Apply the inverse transformation of `y_pipeline` to the output of `wrapped_pipeline`. default is `True`.


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
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
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
        wrapped_pipeline: Pipeline,
        y_pipeline: Union[List[InvertibleTransformation], InvertibleTransformation],
        invert_wrapped_output: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self.wrapped_pipeline = wrap_in_double_list_if_needed(wrapped_pipeline)
        self.y_pipeline = wrap_in_double_list_if_needed(y_pipeline)
        self.invert_wrapped_output = invert_wrapped_output
        self.name = name or "TransformTarget-" + get_concatenated_names(
            self.wrapped_pipeline + self.y_pipeline
        )
        self.metadata = None

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        # TransformTarget's primary transformation transforms `y`, not `X`.
        if y is None:
            return (
                pd.DataFrame(),
                None,
                artifact,
            )  # at inference time, `y` will be None, and we don't need to use primary transformations at all, so we return a dummy DataFrame.
        else:
            return y.to_frame(), None, artifact

    def preprocess_secondary(
        self,
        X: pd.DataFrame,
        y: T,
        artifact: Artifact,
        results_primary: pd.DataFrame,
        index: int,
        fit: bool,
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        return X, to_series(results_primary), artifact

    def postprocess_result_primary(
        self,
        results: List[pd.DataFrame],
        y: Optional[pd.Series],
        original_artifact: Artifact,
        fit: bool,
    ) -> pd.DataFrame:
        return results[0]

    def postprocess_result_secondary(
        self,
        primary_results: pd.DataFrame,
        secondary_results: List[pd.DataFrame],
        y: Optional[pd.Series],
        in_sample: bool,
    ) -> pd.DataFrame:
        if self.invert_wrapped_output is False:
            return secondary_results[0]
        predictions = get_prediction_column(secondary_results[0])
        for transformation in reversed(self.y_pipeline[0]):
            predictions = transformation.inverse_transform(predictions, in_sample)
        results = secondary_results[0]
        results[get_prediction_column_name(results)] = to_series(predictions)
        return results

    def get_children_primary(self) -> Pipelines:
        return self.y_pipeline

    def get_children_secondary(
        self,
    ) -> Optional[Pipelines]:
        return self.wrapped_pipeline

    def clone(self, clone_children: Callable) -> TransformTarget:
        clone = TransformTarget(
            wrapped_pipeline=clone_children(self.wrapped_pipeline),
            y_pipeline=clone_children(self.y_pipeline),
            invert_wrapped_output=self.invert_wrapped_output,
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        clone.id = self.id
        return clone
