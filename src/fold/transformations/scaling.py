# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

from typing import Tuple

from sklearn.preprocessing import MinMaxScaler as SKLearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SKLearnStandardScaler

from fold.transformations.sklearn import WrapInvertibleSKLearnTransformation


class StandardScaler(WrapInvertibleSKLearnTransformation):
    """
    Standardize features by removing the mean and scaling to unit variance.

    A wrapper around SKLearn's StandardScaler.
    Capable of further updates after the initial fit.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import StandardScaler
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = StandardScaler()
    >>> X["sine"].head()
    2021-12-31 07:20:00    0.0000
    2021-12-31 07:21:00    0.0126
    2021-12-31 07:22:00    0.0251
    2021-12-31 07:23:00    0.0377
    2021-12-31 07:24:00    0.0502
    Freq: T, Name: sine, dtype: float64
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> preds["sine"].head()
    2021-12-31 15:40:00   -0.000000
    2021-12-31 15:41:00    0.017819
    2021-12-31 15:42:00    0.035497
    2021-12-31 15:43:00    0.053316
    2021-12-31 15:44:00    0.070994
    Freq: T, Name: sine, dtype: float64

    ```

    References
    ----------

    [SKLearn's StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    """

    name = "StandardScaler"

    def __init__(self):
        super().__init__(SKLearnStandardScaler, init_args=dict())


class MinMaxScaler(WrapInvertibleSKLearnTransformation):
    """
    Transform features by scaling each feature to a given range.

    A wrapper around SKLearn's StandardScaler.
    Capable of further updates after the initial fit.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.

    Examples
    --------
    ```pycon
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.transformations import MinMaxScaler
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> pipeline = MinMaxScaler()
    >>> preds, trained_pipeline = train_backtest(pipeline, X, y, splitter)
    >>> X["sine"].loc[preds.index].head()
    2021-12-31 15:40:00   -0.0000
    2021-12-31 15:41:00    0.0126
    2021-12-31 15:42:00    0.0251
    2021-12-31 15:43:00    0.0377
    2021-12-31 15:44:00    0.0502
    Freq: T, Name: sine, dtype: float64
    >>> preds["sine"].head()
    2021-12-31 15:40:00    0.50000
    2021-12-31 15:41:00    0.50630
    2021-12-31 15:42:00    0.51255
    2021-12-31 15:43:00    0.51885
    2021-12-31 15:44:00    0.52510
    Freq: T, Name: sine, dtype: float64

    ```

    References
    ----------
    [SKLearn's MinMaxScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    """

    name = "MinMaxScaler"

    def __init__(self, feature_range: Tuple[int, int] = (0, 1), clip=False):
        super().__init__(
            SKLearnMinMaxScaler, init_args=dict(feature_range=feature_range, clip=clip)
        )
