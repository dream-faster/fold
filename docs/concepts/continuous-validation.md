
## What is it?

With Continuous Validation, you take an existing time series, and train multiple models, as you simulate your model's performance over time:
![Continuous Validation](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/continous_validation.svg) 

**This way, you can turn almost all of your data into an out of sample test set.**

Instead of only looking at the last 1 year, 1 month of out-of-sample predictions, you can simulate "live deployment" over almost the whole time series.


## How is it different to Time Series Cross-Validation?

Inside a test window, and during deployment, `fold` provides a way for a model to access the last value.
![Continous Online Inference](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/continous_online_inference.svg) 

This way, `fold` blends:
1. the speed of mini-batch Machine learning.
2. with the accuracy of online Machine learning.


## More

The different strategies are implemented with [Splitters](splitters.md) in `fold`.


