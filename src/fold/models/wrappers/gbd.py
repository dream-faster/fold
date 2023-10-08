from __future__ import annotations

from abc import abstractmethod
from math import sqrt
from typing import Any, Callable, Optional, Type, Union

import pandas as pd

from fold.base import Tunable
from fold.models.base import Model

from .types import ClassWeightingStrategy


class WrapGBM(Model, Tunable):
    def __init__(
        self,
        model_class: Type,
        init_args: Optional[dict] = {},
        instance: Optional[Any] = None,
        set_class_weights: Union[
            ClassWeightingStrategy, str
        ] = ClassWeightingStrategy.none,
        params_to_try: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> None:
        self.init_args = init_args
        self.model_class = model_class

        self.model = model_class(**init_args) if instance is None else instance
        self.set_class_weights = ClassWeightingStrategy.from_str(set_class_weights)

        self.properties = Model.Properties(
            requires_X=True, model_type=self.get_model_type(self.model)
        )

        self.name = name or self.model.__class__.__name__
        self.params_to_try = params_to_try

    @abstractmethod
    def get_model_type(self, model) -> Model.Properties.ModelType:
        raise NotImplementedError

    @classmethod
    def from_model(
        cls,
        model,
        set_class_weights: Union[
            ClassWeightingStrategy, str
        ] = ClassWeightingStrategy.none,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ):
        return cls(
            model.__class__,
            init_args=model.get_params(),
            instance=model,
            set_class_weights=set_class_weights,
            name=name,
            params_to_try=params_to_try,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        if self.set_class_weights in [
            ClassWeightingStrategy.balanced,
            ClassWeightingStrategy.balanced_sqrt,
        ]:
            counts = y.value_counts()
            scale_pos_weight = counts[0] / counts[1]
            if self.set_class_weights == ClassWeightingStrategy.balanced_sqrt:
                scale_pos_weight = sqrt(scale_pos_weight)
            self.model = self.model.set_params(
                **dict(scale_pos_weight=scale_pos_weight)
            )
            self.model.fit(
                X=X,
                y=y,
                sample_weight=sample_weights,
            )
        else:
            self.model.fit(X=X, y=y, sample_weight=sample_weights)

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        predictions = pd.Series(self.model.predict(X), index=X.index).rename(
            f"predictions_{self.name}"
        )
        if self.properties.model_type == Model.Properties.ModelType.classifier:
            probabilities = pd.DataFrame(
                data=self.model.predict_proba(X),
                index=X.index,
                columns=[
                    f"probabilities_{self.name}_{float(item)}"
                    for item in self.model.classes_
                ],
            )
            return pd.concat([predictions, probabilities], axis="columns")
        else:
            return predictions

    predict_in_sample = predict

    def get_params(self) -> dict:
        return {
            **self.model.get_params(),
            **dict(set_class_weights=self.set_class_weights.value),
        }

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        if "set_class_weights" in parameters:
            set_class_weights = parameters.pop("set_class_weights")
        return self.__class__(
            self.model_class,
            init_args=parameters,
            set_class_weights=set_class_weights,
            name=self.name,
        )


class WrapLGBM(WrapGBM):
    def get_model_type(self, model) -> Model.Properties.ModelType:
        from lightgbm import LGBMClassifier, LGBMRegressor

        if isinstance(model, LGBMRegressor):
            return Model.Properties.ModelType.regressor
        elif isinstance(model, LGBMClassifier):
            return Model.Properties.ModelType.classifier
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.model.fit(
            X,
            y,
            sample_weight=sample_weights,
            init_model=self.model.get_booster(),
        )


class WrapXGB(WrapGBM):
    def get_model_type(self, model) -> Model.Properties.ModelType:
        from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

        if isinstance(self.model, XGBRegressor) or isinstance(
            self.model, XGBRFRegressor
        ):
            return Model.Properties.ModelType.regressor
        elif isinstance(self.model, XGBClassifier) or isinstance(
            self.model, XGBRFClassifier
        ):
            return Model.Properties.ModelType.classifier
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> None:
        self.model.fit(
            X=X,
            y=y,
            xgb_model=self.model.get_booster(),
            sample_weight=sample_weights,
        )
