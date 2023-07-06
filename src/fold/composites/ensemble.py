# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd

from ..base import Composite, Pipelines, get_concatenated_names
from .columns import postprocess_results


class Ensemble(Composite):
    """
    Ensemble (average) the results of multiple pipelines.

    Parameters
    ----------

    pipelines : Pipelines
        A list of pipelines to be applied to the data, independently of each other.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.composites import Ensemble
        >>> from fold.models import DummyRegressor
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
        >>> pipeline = Ensemble([
        ...     DummyRegressor(0.1),
        ...     DummyRegressor(0.9),
        ... ])
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
        >>> preds.squeeze().head()
        2021-12-31 15:40:00    0.5
        2021-12-31 15:41:00    0.5
        2021-12-31 15:42:00    0.5
        2021-12-31 15:43:00    0.5
        2021-12-31 15:44:00    0.5
        Freq: T, Name: predictions_Ensemble-DummyRegressor-0.1-DummyRegressor-0.9, dtype: float64
    """

    def __init__(self, pipelines: Pipelines, name: Optional[str] = None) -> None:
        self.pipelines = pipelines
        self.name = name or "Ensemble-" + get_concatenated_names(pipelines)
        self.properties = Composite.Properties()
        self.metadata = None

    def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        return postprocess_results(results, self.name)

    def get_children_primary(self) -> Pipelines:
        return self.pipelines

    def clone(self, clone_children: Callable) -> Ensemble:
        clone = Ensemble(
            pipelines=clone_children(self.pipelines),
        )
        clone.properties = self.properties
        clone.name = self.name
        clone.metadata = self.metadata
        return clone
