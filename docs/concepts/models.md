
`Fold` fundamentally supports both:

## "Time series" models

The likes of ARIMA, RNNs, Exponential Smoothing, etc.

Examples we support:

- [StatsForecast](https://github.com/Nixtla/statsforecast/)
- [NeuralForecast](https://github.com/Nixtla/neuralforecast).
- [Prophet](https://facebook.github.io/prophet/)

... 

## Tabular ML models

The likes of Random Forests, Gradient Boosted Trees, Linear Regression, etc.

Examples:

- [XGBoost](https://github.com/dmlc/xgboost)
- [LightGBM](https://github.com/Microsoft/LightGBM/).
... 


**We provide wrappers for various 3rd party libraries in a separate package called [fold-wrapper](https://github.com/dream-faster/fold-wrapper).**

As Time Series is a fundamentally hard problem, it's also important to use strong baselines, which we have our own, fast implementations:

::: fold.models.baseline


In the [Design](design.md) documentation, we explain how `fold` supports both types of models.
