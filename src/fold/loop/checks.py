# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional, Tuple

import pandas as pd


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
    return X, y


def check_types_multi_series(data: pd.DataFrame):
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame."
    assert "y" in data.columns, "data must have a column `y`."
    assert "ds" in data.columns, "data must have a column `ds`."
    assert "unique_id" in data.columns, "data must have a column `unique_id`."
