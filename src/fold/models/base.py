from abc import abstractmethod
from typing import Union

import pandas as pd

from ..transformations.base import Transformation


class Model(Transformation):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Predictions for exclusively out-of-sample data, the model has never seen before.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Predictions for in-sample, already seen data.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if in_sample:
            return postpostprocess_prediction(self.predict_in_sample(X), self.name)
        else:
            return postpostprocess_prediction(self.predict(X), self.name)


def postpostprocess_prediction(
    dataframe_or_series: Union[pd.DataFrame, pd.Series], name: str
) -> pd.DataFrame:
    if isinstance(dataframe_or_series, pd.Series):
        return dataframe_or_series.rename("predictions_" + name).to_frame()
    elif isinstance(dataframe_or_series, pd.DataFrame):
        return dataframe_or_series
    else:
        raise ValueError(
            f"Expected dataframe or series, got {type(dataframe_or_series)} instead."
        )
