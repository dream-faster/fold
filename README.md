<!-- # Fold -->



<p align="center">
  <a href="https://dream-faster.github.io/fold/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/dream-faster/fold/docs.yaml?logo=readthedocs"></a>
  <a href="https://codecov.io/gh/dream-faster/fold" ><img src="https://codecov.io/gh/dream-faster/fold/branch/main/graph/badge.svg?token=Z7I2XSF188"/></a>
  <a href="https://github.com/dream-faster/fold/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/dream-faster/fold/actions/workflows/tests.yaml/badge.svg"/></a>
  <a href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
</p>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://dream-faster.github.io/fold/">
    <img src="https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/logo.png" alt="Logo" width="90" >
  </a>
<h3 align="center"> <i>(/fold/)</i></h3>
  <p align="center">
    Nowcasting with continuous evaluation
    <br />
    <a href="https://dream-faster.github.io/fold/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

**Fold** is a Nowcasting continuous evaluation/deployment library.
(also known as walk-forward evaluation)

It supports both univariate and (soon) multivariate time series.
It is from the ground-up extensible and lightweight.

**Avoid the mistakes people make with time series ML:**

- ignoring useful features otherwise available in production (value in T-1)

- accidentally using information that wouldn't otherwise be available at the time of training/evaluation (lookahead bias)


**It can train models without lookahead bias:**

- with expanding window

- with rolling window

- even with a single train/test split, if you really want it
  
**It can also help you with creating complex blended models:**

- Ensembling: (weighted) averaging the predictions of multiple models or pipelines

- Stacking: feed multiple model's predictions into a model

- Meta-labeling or residual-modelling/hybrid models (coming soon)
... or any combinations of the above.

Why? It [works really well for time series](https://linkinghub.elsevier.com/retrieve/pii/S0169207022001480).
  



  
<br/>

## Fold solves the following problems:

- Time series are often evaluated on single or multiple (finite) train-test splits. This is a source of grave mistakes.<br/>
**→ fold allows to simulate and evaluate your models like they would have performed, in reality/when deployed.**

- Complex models are hard to create and manage<br/>
**→ Fold supports easy composite model creation**

- Too many dependencies and an either you use-all-or-none-of-it approach<br/>
**→ Fold has very few hard dependencies (only pandas, numpy, tqdm and scikit-learn). It supports scikit-learn Pipelines by default. It's main aim is to be as simple and few lines as possible.**

- Works well with industry standard libraries as well as with Myalo's other open source toolkits (eg.: [Krisi evaluation](https://github.com/dream-faster/krisi) or [Fold Models]([h](https://github.com/dream-faster/fold-models))<br/>
**→ Don't need to choose between `sktime`, `darts` or `statsforecast` models. We will or already support them all (coming soon)**

- Most of the time series libraries don't support distributed computing at all.<br/>
**→ Fold was built with distributed computing in mind. Your pipeline is automatically parallelized wherever it can be (coming soon)**

- Easy to deploy.<br/>
**→ Don't stop at training models.**


<br/>

## Installation


The project was entirely built in ``python``. 

Prerequisites

* ``python >= 3.7`` and ``pip``


Install from git directly

*  ``pip install https://github.com/dream-faster/fold/archive/main.zip ``

<br/>

## Quickstart

You can quickly train your chosen models and get predictions by running:

```python
from fold.loop import trian, backtest
X
y = X.squeeze()

splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
transformations = [
  DummyRegressor(strategy="constant", constant=0),
  OnlyPredictions(),
]
transformations_over_time = train(transformations, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)
```




## Features

- Supports both Regression and Classification tasks.
- Online and Mini-batch learning.
- Feature selection and other transformations (coming soon) on an expanding/rolling window basis
- Use any scikit-learn model/pipeline natively!
- Use any univariate or sequence models (wrappers provided in [fold-models](https://github.com/dream-faster/fold-models)).
- Use any Deep Learning Time Series models (wrappers provided in [fold-models](https://github.com/dream-faster/fold-models)).
- Super easy syntax!
- Probabilistic foreacasts (currently, for Classification, soon for Regression as well).
- Hyperparemeter optimization / Model selection (coming soon).


## Limitations

- No intermittent time series support, very limited support for non-continuous time series.
- No multi-step ahead forecasts. If you want to forecast multiple steps ahead, transform `y` to aggregate the change over the forecasting horizon you're interested in.
- No hierarchical time series support.

## Similar libraries
It's like [SKTime](https://github.com/sktime/sktime), but with more focus and 100x less code, designed with distributed computing in mind.
It's like [River](https://github.com/online-ml/river), but supporting mini-batch, not just online learning.
It’s very much like [timemachines](https://github.com/microprediction/timemachines), but with an API that’s more accessible for the Python community.


## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

The project uses ``isort`` and ``black`` for formatting.

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.


