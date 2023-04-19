# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional, Tuple

import pandas as pd


def check_types(
    X: Optional[pd.DataFrame], y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    if X is None:
        X = pd.DataFrame(0, index=y.index, columns=["X_not_available"])
    else:
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame."
    assert isinstance(y, pd.Series), "y must be a pandas Series."
    return X, y
