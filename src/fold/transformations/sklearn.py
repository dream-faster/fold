# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from inspect import getfullargspec
from typing import Optional

import pandas as pd

from ..base import FeatureSelector, InvertibleTransformation, Transformation, fit_noop


class WrapSKLearnTransformation(Transformation):
    """
    Wraps an SKLearn Transformation.
    There's no need to use it directly, `fold` automatically wraps all sklearn transformations into this class.
    """

    properties = Transformation.Properties(requires_X=True)

    def __init__(self, transformation) -> None:
        if hasattr(transformation, "set_output"):
            transformation = transformation.set_output(transform="pandas")
        self.transformation = transformation
        self.name = transformation.__class__.__name__

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
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
    ) -> None:
        if hasattr(self.transformation, "partial_fit"):
            argspec = getfullargspec(self.transformation.partial_fit)
            if len(argspec.args) == 3:
                self.transformation.partial_fit(X, y)
            elif len(argspec.args) == 4:
                self.transformation.partial_fit(X, y, sample_weights)
        # if we don't have partial_fit, we can't update the model (maybe throw an exception, and force user to wrap it into `DontUpdate`?)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if hasattr(self.transformation, "set_output"):
            return self.transformation.transform(X)
        else:
            return pd.DataFrame(self.transformation.transform(X), columns=X.columns)


class WrapInvertibleSKLearnTransformation(
    WrapSKLearnTransformation, InvertibleTransformation
):
    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        return pd.Series(self.transformation.inverse_transform(X), index=X.index)


class WrapSKLearnFeatureSelector(FeatureSelector):
    """
    Wraps an SKLearn Feature Selector class, stores the selected columns in `selected_features` property.
    There's no need to use it directly, `fold` automatically wraps all sklearn feature selectors into this class.
    """

    properties = Transformation.Properties(requires_X=True)
    selected_features: Optional[str] = None

    def __init__(self, transformation) -> None:
        if hasattr(transformation, "set_output"):
            transformation = transformation.set_output(transform="pandas")
        self.transformation = transformation
        self.name = transformation.__class__.__name__

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.transformation.fit(X, y)
        if hasattr(self.transformation, "get_feature_names_out"):
            self.selected_features = self.transformation.get_feature_names_out()
        else:
            self.selected_features = X.columns[
                self.transformation.get_support()
            ].to_list()

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X[self.selected_features]

    update = fit_noop
