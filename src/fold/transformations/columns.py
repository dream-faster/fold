from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..base import Transformation, fit_noop
from ..utils.checks import check_get_columns
from ..utils.list import wrap_in_list


class SelectColumns(Transformation):
    """
    Select a single or multiple columns.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns: List[str] = wrap_in_list(columns)
        self.name = f"SelectColumns-{columns}"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X[self.columns]

    fit = fit_noop
    update = fit


class DropColumns(Transformation):
    """
    Drops a single or multiple columns.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self, columns: Union[List[str], str]) -> None:
        self.columns = wrap_in_list(columns)
        self.name = f"DropColumns-{columns}"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(columns=check_get_columns(self.columns, X))

    fit = fit_noop
    update = fit


class RenameColumns(Transformation):
    """
    Renames columns.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self, columns_mapper: dict) -> None:
        self.columns_mapper = columns_mapper
        self.name = "RenameColumns"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.rename(columns=self.columns_mapper)

    fit = fit_noop
    update = fit


class OnlyPredictions(Transformation):
    """
    Drops all columns except the output model(s)' predictions.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.transformations import OnlyProbabilities
        >>> from fold.models.dummy import DummyClassifier
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
        >>> pipeline = [DummyClassifier(1, [0, 1], [0.5, 0.5]), OnlyPredictions()]
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
        >>> preds.head()
                             predictions_DummyClassifier
        2021-12-31 15:40:00                            1
        2021-12-31 15:41:00                            1
        2021-12-31 15:42:00                            1
        2021-12-31 15:43:00                            1
        2021-12-31 15:44:00                            1

    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self) -> None:
        self.name = "OnlyPredictions"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("predictions_")]
        )

    fit = fit_noop
    update = fit


class OnlyProbabilities(Transformation):
    """
    Drops all columns except the output model(s)' probabilities.

    Examples
    --------
        >>> from fold.loop import train_backtest
        >>> from fold.splitters import SlidingWindowSplitter
        >>> from fold.transformations import OnlyProbabilities
        >>> from fold.models.dummy import DummyClassifier
        >>> from fold.utils.tests import generate_sine_wave_data
        >>> X, y  = generate_sine_wave_data()
        >>> splitter = SlidingWindowSplitter(initial_train_window=0.5, step=0.2)
        >>> pipeline = [DummyClassifier(1, [0, 1], [0.5, 0.5]), OnlyProbabilities()]
        >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
        >>> preds.head()
                             probabilities_DummyClassifier_0  probabilities_DummyClassifier_1
        2021-12-31 15:40:00                              0.5                              0.5
        2021-12-31 15:41:00                              0.5                              0.5
        2021-12-31 15:42:00                              0.5                              0.5
        2021-12-31 15:43:00                              0.5                              0.5
        2021-12-31 15:44:00                              0.5                              0.5

    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self) -> None:
        self.name = "OnlyProbabilities"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("probabilities_")]
        )

    fit = fit_noop
    update = fit
