from typing import Callable, List, Optional

import pandas as pd

from ...splitters import Split
from ...transformations.base import Transformations


def train_transformations(
    func: Callable,
    transformations: Transformations,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    splits: List[Split],
):
    import ray

    func = ray.remote(func)
    X = ray.put(X)
    y = ray.put(y)
    return ray.get(
        [func.remote(X, y, sample_weights, transformations, split) for split in splits]
    )
