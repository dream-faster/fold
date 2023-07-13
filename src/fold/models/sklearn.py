# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from typing import Callable, Optional, Type, Union

import pandas as pd

from ..base import Artifact, Tunable
from .base import Model


class WrapSKLearnClassifier(Model, Tunable):
    """
    Wraps an SKLearn Classifier model.
    There's no need to use it directly, `fold` automatically wraps all sklearn classifiers into this class.
    """

    def __init__(
        self,
        model_class: Type,
        init_args: dict,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.model = model_class(**init_args)
        self.params_to_try = params_to_try
        self.name = name or self.model.__class__.__name__
        self.properties = Model.Properties(
            requires_X=True, model_type=Model.Properties.ModelType.classifier
        )

    @classmethod
    def from_model(
        cls,
        model,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> WrapSKLearnClassifier:
        return cls(
            model_class=model.__class__,
            init_args=model.get_params(),
            params_to_try=params_to_try,
            name=name,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> Optional[Artifact]:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        else:
            self.model.fit(X, y, sample_weights)

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        # if we don't have partial_fit, we can't update the model (maybe throw an exception, and force user to wrap it into `DontUpdate`?)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
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
        return pd.concat([predictions, probabilities], copy=False, axis="columns")

    predict_in_sample = predict

    def get_params(self) -> dict:
        return self.model.get_params()

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        return WrapSKLearnClassifier(
            model_class=self.model.__class__,
            init_args=parameters,
            name=self.name,
        )


class WrapSKLearnRegressor(Model, Tunable):
    """
    Wraps an SKLearn regressor model.
    There's no need to use it directly, `fold` automatically wraps all sklearn regressors into this class.
    """

    def __init__(
        self,
        model_class: Type,
        init_args: dict,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.model = model_class(**init_args)
        self.params_to_try = params_to_try
        self.name = name or self.model.__class__.__name__
        self.properties = Model.Properties(
            requires_X=True, model_type=Model.Properties.ModelType.regressor
        )

    @classmethod
    def from_model(
        cls,
        model,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> WrapSKLearnRegressor:
        return cls(
            model_class=model.__class__,
            init_args=model.get_params(),
            name=name,
            params_to_try=params_to_try,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> Optional[Artifact]:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        else:
            self.model.fit(X, y, sample_weights)

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y, sample_weights)
        # if we don't have partial_fit, we can't update the model (maybe throw an exception, and force user to wrap it into `DontUpdate`?)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return pd.Series(
            data=self.model.predict(X).squeeze(),
            index=X.index,
            name=f"predictions_{self.name}",
        ).to_frame()

    predict_in_sample = predict

    def get_params(self) -> dict:
        return self.model.get_params()

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        return WrapSKLearnRegressor(
            model_class=self.model.__class__,
            init_args=parameters,
            name=self.name,
        )
