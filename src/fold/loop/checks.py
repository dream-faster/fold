# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from logging import warn
from typing import Optional, Tuple

import pandas as pd

from ..base.classes import Extras
from ..utils.trim import trim_initial_nans


def check_types(
    X: Optional[pd.DataFrame], y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    if X is None:
        X = pd.DataFrame(
            pd.arrays.SparseArray(0), index=y.index, columns=["X_not_available"]
        )
    else:
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame."
    assert isinstance(y, pd.Series), "y must be a pandas Series."

    X_trimmed, _, _ = trim_initial_nans(X, y, Extras())
    if len(X_trimmed) < len(X):
        warn(
            f"Detected initial NaNs in X ({ len(X) - len(X_trimmed) } instances),"
            " that'll be trimmed internally. Please trim your data before passing it"
            " to the model."
        )
    return X, y


def check_types_multi_series(data: pd.DataFrame):
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame."
    assert "y" in data.columns, "data must have a column `y`."
    assert "ds" in data.columns, "data must have a column `ds`."
    assert "unique_id" in data.columns, "data must have a column `unique_id`."
