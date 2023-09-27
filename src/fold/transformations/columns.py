# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd

from ..base import Transformation, Tunable, fit_noop
from ..utils.checks import get_list_column_names
from ..utils.list import wrap_in_list


class SelectColumns(Transformation, Tunable):
    """
    Selects a single or multiple columns, drops the rest.

    Parameters
    ----------

    columns : Union[List[str], str]
        The column or columns to select (dropping the rest).


    """

    def __init__(
        self,
        columns: Union[List[str], str],
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.columns: List[str] = wrap_in_list(columns)
        self.params_to_try = params_to_try
        self.name = name or f"SelectColumns-{columns}"
        self.properties = Transformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X[get_list_column_names(self.columns, X)]

    fit = fit_noop
    update = fit


class DropColumns(Transformation, Tunable):
    """
    Drops a single or multiple columns.

    Parameters
    ----------

    columns : List[str], str
        The column or columns to drop.

    """

    def __init__(
        self,
        columns: Union[List[str], str],
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.columns = wrap_in_list(columns)
        self.params_to_try = params_to_try
        self.name = name or f"DropColumns-{columns}"
        self.properties = Transformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(columns=get_list_column_names(self.columns, X), errors="ignore")

    fit = fit_noop
    update = fit


class RenameColumns(Transformation):
    """
    Renames columns.

    Parameters
    ----------

    columns_mapper : dict
        A dictionary containing the old column names as keys and the new column names as values.

    Examples
    --------

    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import RenameColumns
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = RenameColumns({"sine": "sine_renamed"})
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                         sine_renamed
    2021-12-31 15:40:00       -0.0000
    2021-12-31 15:41:00        0.0126
    2021-12-31 15:42:00        0.0251
    2021-12-31 15:43:00        0.0377
    2021-12-31 15:44:00        0.0502

    ```
    """

    def __init__(self, columns_mapper: dict) -> None:
        self.columns_mapper = columns_mapper
        self.name = "RenameColumns"
        self.properties = Transformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.rename(columns=self.columns_mapper)

    fit = fit_noop
    update = fit


class AddColumnSuffix(Transformation):
    """
    Add suffix to column names.

    Parameters
    ----------

    suffix : str
        Suffix to add to all column names.

    Examples
    --------

    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import RenameColumns
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = AddColumnSuffix("_2")
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                         sine_2
    2021-12-31 15:40:00 -0.0000
    2021-12-31 15:41:00  0.0126
    2021-12-31 15:42:00  0.0251
    2021-12-31 15:43:00  0.0377
    2021-12-31 15:44:00  0.0502

    ```
    """

    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        self.name = f"AddColumnSuffix-{self.suffix}"
        self.properties = Transformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.add_suffix(self.suffix)

    fit = fit_noop
    update = fit


class OnlyPredictions(Transformation):
    """
    Drops all columns except the output model(s)' predictions.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import OnlyProbabilities
    >>> from fold.models.dummy import DummyClassifier
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = [DummyClassifier(1, [0, 1], [0.5, 0.5]), OnlyPredictions()]
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                         predictions_DummyClassifier
    2021-12-31 15:40:00                            1
    2021-12-31 15:41:00                            1
    2021-12-31 15:42:00                            1
    2021-12-31 15:43:00                            1
    2021-12-31 15:44:00                            1

    ```
    """

    def __init__(self) -> None:
        self.name = "OnlyPredictions"
        self.properties = Transformation.Properties(requires_X=True)

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
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import OnlyProbabilities
    >>> from fold.models.dummy import DummyClassifier
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = [DummyClassifier(1, [0, 1], [0.5, 0.5]), OnlyProbabilities()]
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds.head()
                         probabilities_DummyClassifier_0  probabilities_DummyClassifier_1
    2021-12-31 15:40:00                              0.5                              0.5
    2021-12-31 15:41:00                              0.5                              0.5
    2021-12-31 15:42:00                              0.5                              0.5
    2021-12-31 15:43:00                              0.5                              0.5
    2021-12-31 15:44:00                              0.5                              0.5

    ```
    """

    def __init__(self) -> None:
        self.name = "OnlyProbabilities"
        self.properties = Transformation.Properties(requires_X=True)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X.drop(
            columns=[col for col in X.columns if not col.startswith("probabilities_")]
        )

    fit = fit_noop
    update = fit
