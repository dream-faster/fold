from typing import Callable, List, Optional

import pandas as pd
from tqdm import tqdm

from ...splitters import Fold
from ...transformations.base import Transformations


def train_transformations(
    func: Callable,
    transformations: Transformations,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    splits: List[Fold],
    never_update: bool,
):
    return [
        func(X, y, sample_weights, transformations, split, never_update)
        for split in tqdm(splits)
    ]
