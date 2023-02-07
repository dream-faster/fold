from typing import Callable, List

import pandas as pd
from tqdm import tqdm

from ...splitters import Split
from ...transformations.base import Transformations


def process_transformations(
    func: Callable,
    transformations: Transformations,
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
):
    return [func(X, y, transformations, split) for split in tqdm(splits)]
