import pandas as pd

from drift.models.base import Model
from drift.transformations.base import Transformation


class SKLearnClassifier(Model):

    properties = Transformation.Properties(
        requires_past_X=False, model_type=Transformation.Properties.ModelType.classifier
    )

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = pd.DataFrame(
            data=self.model.predict_proba(X),
            index=X.index,
            columns=[
                f"probabilities_{self.name}_{item}" for item in self.model.classes_
            ],
        )
        predictions = pd.Series(
            data=self.model.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        )
        return pd.concat([predictions, probabilities], axis=1)


class SKLearnRegressor(Model):

    properties = Transformation.Properties(
        requires_past_X=False, model_type=Transformation.Properties.ModelType.regressor
    )

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.Series(
            data=self.model.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        ).to_frame()
