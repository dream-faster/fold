import pandas as pd


def create_forward_rolling_sum(series: pd.Series, period: int) -> pd.Series:
    assert period > 0
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)
    return series.rolling(window=indexer).sum().shift(-1)
