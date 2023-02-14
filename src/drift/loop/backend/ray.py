from typing import Callable, List

import pandas as pd

from ...splitters import Split
from ...transformations.base import Transformations


def process_transformations(
    func: Callable,
    transformations: Transformations,
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
):
    import ray

    func = ray.remote(func)
    X = ray.put(X)
    y = ray.put(y)
    return ray.get([func.remote(X, y, transformations, split) for split in splits])
