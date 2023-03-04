from random import choices
from typing import List

import numpy as np
import pandas as pd


def generate_sine_wave_data(cycles: int = 2, resolution: int = 1000) -> pd.DataFrame:
    length = np.pi * 2 * cycles
    my_wave = np.sin(np.arange(0, length, length / resolution))
    return pd.Series(
        my_wave,
        name="sine",
        index=pd.date_range(end="2022", periods=len(my_wave), freq="m"),
    ).to_frame()


def generate_all_zeros(length: int = 1000) -> pd.DataFrame:
    return pd.Series(
        [0] * length,
        index=pd.date_range(end="2022", periods=length, freq="m"),
    ).to_frame()


def generate_zeros_and_ones_skewed(
    length: int = 1000, labels=[1, 0], weights: List[float] = [0.2, 0.8]
) -> pd.DataFrame:
    return pd.Series(
        choices(population=labels, weights=weights, k=length),
        index=pd.date_range(end="2022", periods=length, freq="s"),
    ).to_frame()
