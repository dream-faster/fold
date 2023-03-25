from typing import Literal, Optional, Tuple

import pandas as pd


def load_dataset(
    dataset_name: str,
    base_path: str = "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets",
) -> pd.DataFrame:
    return pd.read_csv(f"{base_path}/{dataset_name}.csv", parse_dates=True, index_col=0)


def process_dataset(
    df: pd.DataFrame,
    target_col: str,
    deduplicate_strategy: Optional[Literal["first", "last", False]] = None,
    shorten: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if deduplicate_strategy is not None:
        df = df[~df.index.duplicated(keep=deduplicate_strategy)]
    if shorten is not None:
        df = df[:shorten]
    y = df[target_col].shift(-1)[:-1]
    X = df[:-1]

    return X, y


def get_preprocessed_dataset(
    dataset_name: str,
    target_col: str,
    base_path: str = "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets",
    deduplicate_strategy: Optional[Literal["first", "last", False]] = None,
    shorten: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    return process_dataset(
        load_dataset(dataset_name, base_path),
        target_col=target_col,
        deduplicate_strategy=deduplicate_strategy,
        shorten=shorten,
    )
