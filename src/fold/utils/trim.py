# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

import logging
from typing import TypeVar

import pandas as pd

T = TypeVar("T", pd.Series, pd.Series | None)
logger = logging.getLogger("fold:utils")


def get_first_valid_index(series: pd.Series | pd.DataFrame) -> int:
    if series.empty:
        return 0
    if isinstance(series, pd.DataFrame):
        return next(
            (
                idx
                for idx, (_, x) in enumerate(series.iterrows())
                if not pd.isna(x).any()
            ),
            None,
        )
    if isinstance(series, pd.Series):
        return next(
            (idx for idx, (_, x) in enumerate(series.items()) if not pd.isna(x)),
            None,
        )
    return None
