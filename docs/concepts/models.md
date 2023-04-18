
## Model types (Time series / Tabular)

`Fold` fundamentally supports both:

### "Time series" models

The likes of ARIMA, RNNs, Exponential Smoothing, etc.

Their univariate variations only have access to `y`, and ignore all data in `X`.
They're usually designed to be effective without additional feature engineering.

Examples:

- [StatsForecast](https://github.com/Nixtla/statsforecast/)
- [NeuralForecast](https://github.com/Nixtla/neuralforecast)
- [Prophet](https://facebook.github.io/prophet/)

... provided in [fold-wrappers](https://github.com/dream-faster/fold-wrappers).


### Tabular ML models

The likes of Random Forests, Gradient Boosted Trees, Linear Regression, etc.

They depend on having `X` populated, and do not work as "univariate" models.
Each row in `X` corresponds to a single dependent variable, in `y`.

Usually, you may want to add lagged values of `y` with the [AddLagsY][fold.transformations.lags.AddLagsY] class, or create other features for the tabular models with:

- [AddLagsX][fold.transformations.lags.AddLagsX]: if you have exogenous data already.
- [AddWindowFeatures][fold.transformations.window.AddWindowFeatures]: if you have exogenous data already, and you want to aggregate them across different windows.


Examples:

- [Scikit-Learn](http://scikit-learn.org/)
- [XGBoost](https://github.com/dmlc/xgboost)
- [LightGBM](https://github.com/Microsoft/LightGBM/)

... provided in [fold-wrappers](https://github.com/dream-faster/fold-wrappers).

Check out the [Examples gallery](/examples/) to see how easy it is to engineer features with `fold`.


## Online and Mini-batch Learning Modes

A mini-batch model is retrained for every split the [Splitter](splitters.md) returns.
It can not update its state within a test window, but it may depend on lagged values of `X` or `y`.

An `online` model, on the other hand is updated after inference on each timestamp.
Except for "in sample" predictions, which is done in a batch manner, with `predict_in_sample()` 

![Continous Online Inference](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/continous_online_inference.svg) 

We also give our "online" models a way to access the latest values and skip the step that'd update their parameters. This enables an efficient "quasi-online" behaviour, where the model is only re-trained (or, updated) once per fold, but can "follow" the time series data - which usually comes with signifcant increase in accuracy.


## Baselines

As Time Series is a fundamentally hard problem, it's also important to use strong baselines, which we have our own, fast implementations, in [fold-models](https://github.com/dream-faster/fold-wrappers).
