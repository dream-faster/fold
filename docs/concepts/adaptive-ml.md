## What is Adaptive ML?

Time series models that are updated often outperform their static counterparts.
You can almost always improve your model's performance with an simple update, by continuously training them as data comes in.


## What is Adaptive Backtesting?

With Adaptive Backtesting, you take an existing time series, and train multiple models, as you simulate your model's performance over time:
![Adaptive Backtesting](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/continous_validation.svg) 

**This way, you can turn almost all of your data into an out of sample test set.**

Instead of only looking at the last 1 year, 1 month of out-of-sample predictions, you can simulate "live deployment" over almost the whole time series.


## How is it different to Time Series Cross-Validation?

Inside a test window, and during deployment, `fold` provides a way for a model to access the last value.
![Continous Online Inference](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/continous_online_inference.svg) 

This way, `fold` blends:

1. the speed of (Mini-)batch Machine Learning.

2. with the accuracy of [Adaptive (or Online) Machine learning](https://en.wikipedia.org/wiki/Online_machine_learnings). 



## What's wrong with classical Backtesting ("Time Series Cross-Validation")?

A simple example with the `Naive` model (that just repeats the last value):

**Classical Backtesting**

![Time Series Cross-Validation with Naive model](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/naive_cross_validation.png)

The model is static, and repeats the last value for the whole test window.

**Adaptive Backtesting**

![Time Series Cross-Validation with Naive model](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/naive_continuous_validation.png)

The model is up-to-date, and repeats the last value, as you'd expect.

## How is that implemented?

There are two ways to have a model that's up-to-date:

1. The model parameters are updated on each timestamp, within the test window. This can be **really slow**, but a widely available option. See the [Speed Comparison](/concepts/speed) section.

2. Give the model special access to the last value, while keeping its parameters constant. This is what `fold` does with all of our models. This means that the model is still only trained once per fold, providing an order of magnitude speedup, compared to the first method.



## More

The different strategies are implemented with [Splitters](splitters.md) in `fold`.


