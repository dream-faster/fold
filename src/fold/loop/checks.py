from typing import Optional, Tuple

import pandas as pd


def check_types(
    X: Optional[pd.DataFrame], y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    if X is None:
        X = pd.DataFrame(0, index=y.index, columns=[0])
    else:
        assert type(X) is pd.DataFrame, "X must be a pandas DataFrame."
    assert type(y) is pd.Series, "y must be a pandas Series."
    return X, y
