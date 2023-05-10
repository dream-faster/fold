# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from fold.composites.common import get_concatenated_names
from fold.utils.list import wrap_in_double_list_if_needed

from ..base import Composite, Pipeline, Pipelines, T, V


class Sample(Composite):
    """
    Sample data with an imbalanced-learn sampler instance during training.
    No sampling is done during inference or backtesting.

    Warning:
    This seriously challenges the continuity of the data, which is very important for traditional time series models.
    Use with caution, and only with tabular ML models.

    Parameters
    ----------
    sampler : Any
        An imbalanced-learn sampler instance (subclass of `BaseSampler`).
    pipeline : Pipeline
        A pipeline to be applied to the sampled data.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.composites import ModelResiduals
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from imblearn.under_sampling import RandomUnderSampler
        >>> from fold.utils.tests import generate_zeros_and_ones_skewed
        >>> X, y  = generate_zeros_and_ones_skewed()
        >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
        >>> pipeline = Sample(
        ...     sampler=RandomUnderSampler(),
        ...     pipeline=RandomForestClassifier(),
        ... )
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)


    References
    ----------
    [imbalanced-learn](https://imbalanced-learn.org/)
    """

    properties = Composite.Properties(
        primary_requires_predictions=False,
        primary_only_single_pipeline=True,
    )

    def __init__(
        self,
        sampler: Any,
        pipeline: Pipeline,
    ) -> None:
        self.sampler = sampler
        self.pipeline = wrap_in_double_list_if_needed(pipeline)
        self.name = f"Sample-{sampler.__class__.__name__}-{get_concatenated_names(self.pipeline)}"
        super().__init__()

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, sample_weights: V, fit: bool
    ) -> Tuple[pd.DataFrame, T, V]:
        if fit:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            X_resampled.columns = X.columns
            if y is not None:
                y_resampled.name = y.name
            sample_weights_resampled = (
                sample_weights.iloc[self.sampler.sample_indices_]
                if sample_weights is not None
                else None
            )
            return X_resampled, y_resampled, sample_weights_resampled
        else:
            return X, y, sample_weights

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return results[0]

    def get_child_transformations_primary(self) -> Pipelines:
        return self.pipeline

    def clone(self, clone_children: Callable) -> Sample:
        clone = Sample(
            sampler=self.sampler,
            pipeline=clone_children(self.pipeline),
        )
        clone.properties = self.properties
        return clone
