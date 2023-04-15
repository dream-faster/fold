
`Fold` fundamentally supports both:

## "Time series" models

The likes of ARIMA, RNNs, Exponential Smoothing, etc.

Their univariate variations only have access to `y`, and ignore all data in `X`.
They're usually designed to be effective without additional feature engineering.

Examples:

- [StatsForecast](https://github.com/Nixtla/statsforecast/)
- [NeuralForecast](https://github.com/Nixtla/neuralforecast)
- [Prophet](https://facebook.github.io/prophet/)

... provided in [fold-wrapper](https://github.com/dream-faster/fold-wrapper).


## Tabular ML models

The likes of Random Forests, Gradient Boosted Trees, Linear Regression, etc.

They depend on having `X` populated, and do not work as "univariate" models.
Each row in `X` corresponds to a single dependent variable, in `y`.

It's useful to add lagged values of `y` with the [AddLagsY][fold.transformations.lags.AddLagsY] Transformation.
Other straightforward options to create new features for the tabular models are:

- [AddLagsX][fold.transformations.lags.AddLagsX] if you have exogenous data already.
- [AddWindowFeatures][fold.transformations.window.AddWindowFeatures] if you have exogenous data already, and you want to aggregate them across different windows.


Examples:

- [Scikit-Learn](http://scikit-learn.org/)
- [XGBoost](https://github.com/dmlc/xgboost)
- [LightGBM](https://github.com/Microsoft/LightGBM/)

... provided in [fold-wrapper](https://github.com/dream-faster/fold-wrapper).


## Baselines

As Time Series is a fundamentally hard problem, it's also important to use strong baselines, which we have our own, fast implementations:

::: fold.models.baseline


In the [Design](design.md) documentation, we explain how `fold` supports both types of models.
