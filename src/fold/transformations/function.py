# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from collections.abc import Callable

import pandas as pd

from ..base import Transformation, Tunable, fit_noop
from ..utils.dataframe import apply_function_batched, fill_na_inf


class ApplyFunction(Transformation, Tunable):
    """
    Wraps and arbitrary function that will run at inference.
    """

    def __init__(
        self,
        func: Callable,
        past_window_size: int | None,
        fillna: bool = False,
        batch_columns: int | None = None,
        output_dtype: type | None = None,
        name: str | None = None,
        params_to_try: dict | None = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.fillna = fillna
        self.past_window_size = past_window_size
        self.batch_columns = batch_columns
        self.output_dtype = output_dtype
        self.params_to_try = params_to_try
        self.properties = Transformation.Properties(
            requires_X=True, memory_size=past_window_size, disable_memory=True
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        def convert_dtype_if_needed(df: pd.DataFrame) -> pd.DataFrame:
            return df.astype(self.output_dtype) if self.output_dtype is not None else df

        return_value = convert_dtype_if_needed(
            apply_function_batched(X, self.func, self.batch_columns)
        )
        return fill_na_inf(return_value) if self.fillna else return_value

    fit = fit_noop
    update = fit
