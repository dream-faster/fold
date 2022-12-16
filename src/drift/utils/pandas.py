import pandas as pd


def shift_and_duplicate_first_value(series: pd.Series, n: int) -> pd.Series:
    series = series.shift(n)
    series.iloc[:n] = series.iloc[n]
    return series
