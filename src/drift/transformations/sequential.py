from .base import Transformation

import pandas as pd
from typing import TYPE_CHECKING


if TYPE_CHECKING: 
    from neuralforecast.common._base_windows import BaseWindows


class SequentialTransformation(Transformation):

    properties = Transformation.Properties(requires_past_X=True)

    def __init__(self, transformation: "BaseWindows") -> None:
        from neuralforecast import NeuralForecast
        self.transformation = NeuralForecast(models=[transformation], freq='M')
        
        self.name = self.transformation.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Y_df = X.copy(deep=True)
        Y_df['unique_id'] = 1.
        # Y_df['y'] = X[X.columns[0]]
        
        # Y_df = Y_df.rename(columns={'timestamp': 'ds'})
        Y_df = Y_df[['unique_id', 'ds', 'y']]
        
        self.transformation.fit(Y_df)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.transformation.predict().reset_index(), columns=X.columns)
