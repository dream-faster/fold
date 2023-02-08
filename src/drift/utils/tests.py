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


def generate_all_zeros(length: int) -> pd.DataFrame:
    return pd.Series(
        [0] * length,
        index=pd.date_range(end="2022", periods=length, freq="m"),
    ).to_frame()
