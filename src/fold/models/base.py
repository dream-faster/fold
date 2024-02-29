# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from abc import abstractmethod

import pandas as pd

from ..base import Transformation
from ..utils.checks import (
    get_classes_from_probabilies_column_names,
    get_probabilities_column_names,
    has_probabilities,
)


class Model(Transformation):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Predictions for exclusively out-of-sample data, the model has never seen before.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_in_sample(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Predictions for in-sample, already seen data.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if in_sample:
            return postpostprocess_output(self.predict_in_sample(X), self.name)
        return postpostprocess_output(self.predict(X), self.name)


def postpostprocess_output(
    dataframe_or_series: pd.DataFrame | pd.Series, name: str
) -> pd.DataFrame:
    if isinstance(dataframe_or_series, pd.Series):
        return dataframe_or_series.rename(f"predictions_{name}").to_frame()
    if isinstance(dataframe_or_series, pd.DataFrame):
        if len(dataframe_or_series.columns) == 1:
            dataframe_or_series.columns = [f"predictions_{name}"]
            return dataframe_or_series
        if has_probabilities(dataframe_or_series):
            # Sort probabilities columns if necessary
            prob_columns = get_probabilities_column_names(dataframe_or_series)
            prob_columns_sorted = sorted(
                get_classes_from_probabilies_column_names(prob_columns)
            )
            if prob_columns == prob_columns_sorted:
                return dataframe_or_series
            return dataframe_or_series.sort_index(axis="columns")
        return dataframe_or_series
    raise ValueError(
        f"Expected dataframe or series, got {type(dataframe_or_series)} instead."
    )
