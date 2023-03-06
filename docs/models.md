
`Fold` fundamentally supports both:

- Tabular learning models, the likes of [XGBoost](https://github.com/dmlc/xgboost) and [LightGBM](https://github.com/Microsoft/LightGBM/)
- Sequence models, the likes of [statsforecast](https://github.com/Nixtla/statsforecast/) or [neuralforecast](https://github.com/Nixtla/neuralforecast)

**We provide wrappers for various 3rd party libraries in a separate package called [fold-models](https://github.com/dream-faster/fold-models).**

As Time Series is a fundamentally hard problem, it's also important to use strong baselines, which we have our own, fast implementations:

::: fold.models.baseline


In the [Design](design.md) documentation, we explain how `fold` supports both types of models.
