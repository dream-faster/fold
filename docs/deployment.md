
When you feel like you have a great model, then you have an option:

### Train a new model, from scratch
Warning: Does not support "SlidingWindowSplitter".

```
    deployable_transformations = train_for_deployment(transformations, X, y)
```

This takes all the data, and uses them to produce a deployable Transformations.


### Use your existing, backtested Models in production

Coming soon


## Inference

```
    first_prediction = infer(
        deployable_transformations,
        new_data,
    )
```

Warning: `infer()` does not mutate the Transformations.
Please make sure that after each prediction, when the ground truth (`y`) becomes available, you update the Models:
```
    deployable_transformations = update(deployable_transformations, X, y)
```

This is only important when you have any kind of sequence, or univariate models in production, like a Seasonal ARIMA that needs to "understand" where it is in time.

