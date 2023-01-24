from typing import List

import pandas as pd
from sklearn.linear_model import LinearRegression

from drift.models.base import Model
from drift.utils.list import shift


class AR(Model):
    def __init__(self, p: int) -> None:
        self.p = p
        self.name = f"AR-{str(p)}"
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X.to_frame(), y.squeeze())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.DataFrame(self.model.predict(X.to_frame()), index=X.index)


class SAR(Model):
    def __init__(self, p: int, season_length: int) -> None:
        self.p = p
        self.name = f"SAR-{str(p)}-{str(season_length)}"
        self.models = [LinearRegression() for _ in range(season_length)]
        self.season_length = season_length

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        seasonal_X_y = [
            (
                X.shift(season).to_frame().values[:: self.season_length],
                y.shift(season).values[:: self.season_length],
            )
            for season in range(self.season_length)
        ]
        for index, local_X_y in enumerate(seasonal_X_y):
            local_X = local_X_y[0]
            local_X[0] = local_X[1]
            local_y = local_X_y[1]
            local_y[0] = local_y[1]
            self.models[index].fit(local_X, local_X_y[1])
        self.models = shift(self.models, len(X) % self.season_length)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = [
            self.models[index % self.season_length].predict([[item]])
            for index, item in enumerate(X)
        ]
        return pd.DataFrame(preds, index=X.index)
