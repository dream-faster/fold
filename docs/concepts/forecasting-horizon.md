# There's no explicit of `forecasting horizon` in fold.

`fold` is fundamentally specialized in single-step ahead (short-term, high frequency) forecasts.

While we support models that are built for multi-step ahead foreacsting, the support is implicit: it's the `step_size` you set on a [Splitter](/concepts/splitters). Please see [this walkthrough](walkthroughs/benchmarking_sktime_fold/) to gain an intuition on how this could be done.

To get better long-term forecasts, you can also resample your data, or transform your target (`y`) to be an aggregate of the next day, week, month or year's values. Based on our experience, you're better off using different models for different time horizons.

See a detailed explanation [of the different kind of models (and their behaviour) that are available in `fold` here.](/concepts/models/#online-and-mini-batch-learning-models)