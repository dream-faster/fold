# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import pandas as pd

from fold.base.classes import Artifact, Sampler

from ..base import Pipeline, Pipelines, T, get_concatenated_names
from ..utils.list import wrap_in_double_list_if_needed


class Sample(Sampler):
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

    def __init__(
        self, sampler: Any, pipeline: Pipeline, name: Optional[str] = None
    ) -> None:
        self.sampler = sampler

        self.pipeline = wrap_in_double_list_if_needed(pipeline)
        self.name = (
            name
            or f"Sample-{sampler.__class__.__name__}-{get_concatenated_names(self.pipeline)}"
        )

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, artifact: Artifact, fit: bool
    ) -> Tuple[pd.DataFrame, T, Artifact]:
        if fit:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            X_resampled.columns = X.columns
            if y is not None:
                y_resampled.name = y.name
            artifact_resampled = artifact.iloc[self.sampler.sample_indices_]
            artifact_resampled.index = X_resampled.index
            return X_resampled, y_resampled, artifact_resampled
        else:
            return X, y, artifact

    def get_children_primary(self) -> Pipelines:
        return self.pipeline

    def clone(self, clone_children: Callable) -> Sample:
        clone = Sample(
            sampler=self.sampler,
            pipeline=clone_children(self.pipeline),
        )
        clone.name = self.name
        clone.id = self.id
        return clone
