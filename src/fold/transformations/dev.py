# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from inspect import getfullargspec
from typing import Callable, Optional

import pandas as pd

from ..base import Artifact, InvertibleTransformation, Transformation, fit_noop


class Breakpoint(Transformation):
    """
    A transformation that stops execution at the specified point.
    """

    name = "Breakpoint"

    def __init__(
        self,
        stop_at_fit: bool = True,
        stop_at_update: bool = True,
        stop_at_transform: bool = True,
    ) -> None:
        self.stop_at_fit = stop_at_fit
        self.stop_at_update = stop_at_update
        self.stop_at_transform = stop_at_transform
        self.properties = Transformation.Properties(requires_X=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        if self.stop_at_fit:
            breakpoint()

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        if self.stop_at_update:
            breakpoint()

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.stop_at_transform:
            breakpoint()
        return X


class Identity(InvertibleTransformation):
    def __init__(
        self,
        name: Optional[str] = None,
    ) -> None:
        self.name = name or "Identity"
        self.properties = Transformation.Properties(requires_X=False)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X

    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        return X

    fit = fit_noop
    update = fit


class Test(InvertibleTransformation):
    __test__ = False
    no_of_calls_fit = 0
    no_of_calls_update = 0
    no_of_calls_transform_insample = 0
    no_of_calls_transform_outofsample = 0
    no_of_calls_inverse_transform = 0
    name = "Test"

    def __init__(
        self,
        fit_func: Callable,
        transform_func: Callable,
        update_func: Optional[Callable] = None,
        inverse_transform_func: Optional[Callable] = None,
        memory_size: Optional[int] = None,
    ) -> None:
        self.fit_func = fit_func
        self.transform_func = transform_func
        self.update_func = update_func
        self.inverse_transform_func = inverse_transform_func
        self.properties = InvertibleTransformation.Properties(
            requires_X=False, memory_size=memory_size
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.no_of_calls_fit += 1
        self.input_columns_fit = X.columns.to_list()

        argspec = getfullargspec(self.fit_func)
        if len(argspec.args) == 1:
            self.fit_func(X)
        elif len(argspec.args) == 2:
            self.fit_func(X, y)
        elif len(argspec.args) == 3:
            self.fit_func(X, y, sample_weights)
        else:
            raise ValueError(
                "fit_func must accept between 1 and 3 arguments, "
                f"but {len(argspec.args)} were given."
            )

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.no_of_calls_update += 1
        self.input_columns_update = X.columns.to_list()

        if self.update_func is None:
            return

        argspec = getfullargspec(self.update_func)
        if len(argspec.args) == 1:
            self.update_func(X)
        elif len(argspec.args) == 2:
            self.update_func(X, y)
        elif len(argspec.args) == 3:
            self.update_func(X, y, sample_weights)
        else:
            raise ValueError(
                "update_func must accept between 1 and 3 arguments, "
                f"but {len(argspec.args)} were given."
            )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        self.input_columns_transform = X.columns.to_list()

        if in_sample:
            self.no_of_calls_transform_insample += 1
        else:
            self.no_of_calls_transform_outofsample += 1

        argspec = getfullargspec(self.transform_func)
        if len(argspec.args) == 1:
            return_value = self.transform_func(X)
        elif len(argspec.args) == 2:
            return_value = self.transform_func(X, in_sample)
        if return_value is None:
            return X
        return return_value

    def inverse_transform(self, X: pd.Series, in_sample: bool) -> pd.Series:
        self.no_of_calls_inverse_transform += 1
        if self.inverse_transform_func is not None:
            return self.inverse_transform_func(X)
        return X


class Lookahead(Transformation):
    """
    A transformation that stops execution at the specified point.
    """

    name = "Lookahead"
    properties = Transformation.Properties(
        requires_X=False,
        mode=Transformation.Properties.Mode.online,
        memory_size=1,
        _internal_supports_minibatch_backtesting=True,
    )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return self._state.memory_y.to_frame()

    fit = fit_noop
    update = fit
