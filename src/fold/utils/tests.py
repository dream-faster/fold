from random import choices
from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_sine_wave_data(
    cycles: int = 2, resolution: int = 1000
) -> Tuple[pd.DataFrame, pd.Series]:
    resolution += 1
    length = np.pi * 2 * cycles
    my_wave = np.sin(np.arange(0, length, length / resolution))
    series = pd.Series(
        my_wave,
        name="sine",
        index=pd.date_range(end="2022", periods=len(my_wave), freq="m"),
    )
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y


def generate_all_zeros(length: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    length += 1
    series = pd.Series(
        [0] * length,
        index=pd.date_range(end="2022", periods=length, freq="m"),
    )
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y


def generate_zeros_and_ones_skewed(
    length: int = 1000, labels=[1, 0], weights: List[float] = [0.2, 0.8]
) -> Tuple[pd.DataFrame, pd.Series]:
    length += 1
    series = pd.Series(
        choices(population=labels, weights=weights, k=length),
        index=pd.date_range(end="2022", periods=length, freq="s"),
    )
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y


def generate_monotonous_data(length: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    length += 1
    increment = 1 / length
    values = np.arange(0, 1, increment)
    series = pd.Series(
        values,
        name="linear",
        index=pd.date_range(end="2022", periods=len(values), freq="m"),
    )
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y
