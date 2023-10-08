# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from __future__ import annotations

from inspect import getfullargspec
from typing import Callable, Optional, Type

import numpy as np
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

    def __init__(
        self,
        transformation_class: Type,
        init_args: dict,
        output_dtype: Optional[type] = None,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.transformation_class = transformation_class
        self.init_args = init_args
        self.transformation = transformation_class(**init_args)
        if hasattr(self.transformation, "set_output"):
            self.transformation = self.transformation.set_output(transform="pandas")
        self.params_to_try = params_to_try
        self.output_dtype = output_dtype
        self.name = name or self.transformation.__class__.__name__
        self.properties = Transformation.Properties(requires_X=True)

    @classmethod
    def from_model(
        cls,
        model,
        output_dtype: Optional[type] = None,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> WrapSKLearnTransformation:
        return cls(
            transformation_class=model.__class__,
            init_args=model.get_params(deep=False),
            output_dtype=output_dtype,
            name=name,
            params_to_try=params_to_try,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> Optional[Artifact]:
        fit_func = (
            self.transformation.partial_fit
            if hasattr(self.transformation, "partial_fit")
            else self.transformation.fit
        )

        argspec = getfullargspec(fit_func)
        if len(argspec.args) == 2 or len(argspec.args) == 1:
            fit_func(X)
        elif len(argspec.args) == 3:
            fit_func(X, y)
        elif len(argspec.args) == 4:
            fit_func(X, y, sample_weights)
        else:
            raise ValueError(
                f"Unexpected number of arguments in {self.transformation.__class__.__name__}.fit"
            )

    def update(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> Optional[Artifact]:
        if not hasattr(self.transformation, "partial_fit"):
            return
        argspec = getfullargspec(self.transformation.partial_fit)

        if len(argspec.args) == 2:
            self.transformation.partial_fit(X)
        elif len(argspec.args) == 3:
            self.transformation.partial_fit(X, y)
        elif len(argspec.args) == 4:
            self.transformation.partial_fit(X, y, sample_weights)
        else:
            raise ValueError(
                f"Unexpected number of arguments in {self.transformation.__class__.__name__}.partial_fit"
            )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        def convert_dtype_if_needed(df: pd.DataFrame) -> pd.DataFrame:
            return df.astype(self.output_dtype) if self.output_dtype is not None else df

        result = convert_dtype_if_needed(self.transformation.transform(X))
        if hasattr(self.transformation, "set_output"):
            return result
        else:
            if result.shape[1] != len(X.columns):
                columns = [f"{self.name}_{i}" for i in range(result.shape[1])]
            else:
                columns = X.columns
            return pd.DataFrame(result, index=X.index, columns=columns)

    def get_params(self) -> dict:
        return self.transformation.get_params(deep=False)

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        return WrapSKLearnTransformation(
            transformation_class=self.transformation.__class__,
            output_dtype=self.output_dtype,
            init_args=parameters,
            name=self.name,
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

    selected_features: Optional[str] = None

    def __init__(
        self,
        transformation_class: Type,
        init_args: dict,
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.transformation = transformation_class(**init_args)
        self.transformation_class = transformation_class
        self.init_args = init_args
        if hasattr(self.transformation, "set_output"):
            self.transformation = self.transformation.set_output(transform="pandas")
        self.params_to_try = params_to_try
        self.name = name or self.transformation.__class__.__name__
        self.properties = Transformation.Properties(requires_X=True)

    @classmethod
    def from_model(
        cls, model, name: Optional[str] = None, params_to_try: Optional[dict] = None
    ) -> WrapSKLearnFeatureSelector:
        return cls(
            transformation_class=model.__class__,
            init_args=model.get_params(deep=False),
            name=name,
            params_to_try=params_to_try,
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> Optional[Artifact]:
        self.transformation.fit(X, y)
        if hasattr(self.transformation, "get_feature_names_out"):
            self.selected_features = self.transformation.get_feature_names_out()
        else:
            self.selected_features = X.columns[
                self.transformation.get_support()
            ].to_list()
        return pd.DataFrame(
            {f"selected_features_{self.name}": [self.selected_features]},
            index=X.index[-1:],
        ).reindex(X.index)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X[self.selected_features]

    def get_params(self) -> dict:
        return self.transformation.get_params(deep=False)

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        return WrapSKLearnFeatureSelector(
            transformation_class=self.transformation.__class__,
            init_args=parameters,
            name=self.name,
        )

    update = fit_noop


class RemoveLowVarianceFeatures(FeatureSelector):
    def __init__(self, threshold: float = 1e-5, name: Optional[str] = None) -> None:
        self.threshold = threshold
        self.properties = Transformation.Properties(requires_X=True)
        self.name = name or "RemoveLowVarianceFeatures"

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> Optional[Artifact]:
        self.variances_ = np.nanvar(X, axis=0)
        if self.threshold == 0:
            peak_to_peaks = np.ptp(X, axis=0)
            compare_arr = np.array([self.variances_, peak_to_peaks])
            self.variances_ = np.nanmin(compare_arr, axis=0)

        if np.all(~np.isfinite(self.variances_) | (self.variances_ <= self.threshold)):
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))
        mask = self.variances_ > self.threshold
        self.selected_features = X.columns[mask].to_list()

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X[self.selected_features]

    update = fit_noop
