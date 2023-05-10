# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from inspect import getfullargspec
from typing import Callable, Optional, Tuple, Type

import pandas as pd

from ..base import (
    Artifact,
    FeatureSelector,
    InvertibleTransformation,
    Transformation,
    Tunable,
    fit_noop,
)


class WrapSKLearnTransformation(Transformation, Tunable):
    """
    Wraps an SKLearn Transformation.
    There's no need to use it directly, `fold` automatically wraps all sklearn transformations into this class.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(
        self,
        transformation_class: Type,
        init_args: dict,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.transformation = transformation_class(**init_args)
        if hasattr(self.transformation, "set_output"):
            self.transformation = self.transformation.set_output(transform="pandas")
        self.name = self.transformation.__class__.__name__
        self.params_to_try = params_to_try

    @classmethod
    def from_model(
        cls,
        model,
        params_to_try: Optional[dict] = None,
    ) -> WrapSKLearnTransformation:
        return cls(
            transformation_class=model.__class__,
            init_args=model.get_params(),
            params_to_try=params_to_try,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        fit_func = (
            self.transformation.partial_fit
            if hasattr(self.transformation, "partial_fit")
            else self.transformation.fit
        )

        argspec = getfullargspec(fit_func)
        if len(argspec.args) == 3:
            fit_func(X, y)
        elif len(argspec.args) == 4:
            fit_func(X, y, sample_weights)

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        if hasattr(self.transformation, "partial_fit"):
            argspec = getfullargspec(self.transformation.partial_fit)
            if len(argspec.args) == 3:
                self.transformation.partial_fit(X, y)
            elif len(argspec.args) == 4:
                self.transformation.partial_fit(X, y, sample_weights)
        # if we don't have partial_fit, we can't update the model (maybe throw an exception, and force user to wrap it into `DontUpdate`?)

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        if hasattr(self.transformation, "set_output"):
            return self.transformation.transform(X), None
        else:
            return (
                pd.DataFrame(self.transformation.transform(X), columns=X.columns),
                None,
            )

    def get_params(self) -> dict:
        return self.transformation.get_params()

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        return WrapSKLearnTransformation(
            transformation_class=self.transformation.__class__,
            init_args=parameters,
        )


class WrapInvertibleSKLearnTransformation(
    WrapSKLearnTransformation, InvertibleTransformation
):
    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        return pd.Series(self.transformation.inverse_transform(X), index=X.index)


class WrapSKLearnFeatureSelector(FeatureSelector, Tunable):
    """
    Wraps an SKLearn Feature Selector class, stores the selected columns in `selected_features` property.
    There's no need to use it directly, `fold` automatically wraps all sklearn feature selectors into this class.
    """

    properties = Transformation.Properties(requires_X=True)
    selected_features: Optional[str] = None

    def __init__(
        self,
        transformation_class: Type,
        init_args: dict,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.transformation = transformation_class(**init_args)
        if hasattr(self.transformation, "set_output"):
            self.transformation = self.transformation.set_output(transform="pandas")
        self.name = self.transformation.__class__.__name__
        self.params_to_try = params_to_try

    @classmethod
    def from_model(
        cls,
        model,
        params_to_try: Optional[dict] = None,
    ) -> WrapSKLearnFeatureSelector:
        return cls(
            transformation_class=model.__class__,
            init_args=model.get_params(),
            params_to_try=params_to_try,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.transformation.fit(X, y)
        if hasattr(self.transformation, "get_feature_names_out"):
            self.selected_features = self.transformation.get_feature_names_out()
        else:
            self.selected_features = X.columns[
                self.transformation.get_support()
            ].to_list()
        return pd.DataFrame(
            {"selected_features": [self.selected_features]}, index=X.index[-1:]
        )

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        return X[self.selected_features], None

    def get_params(self) -> dict:
        return self.transformation.get_params()

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        return WrapSKLearnFeatureSelector(
            transformation_class=self.transformation.__class__,
            init_args=parameters,
        )

    update = fit_noop
