<!-- # Fold -->
<!-- Formatted with Prettier -->

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
<h3 align="center"><b>FOLD</b><br> <i>(/fold/)</i></h3>
  <p align="center">
    <b>[Now|Fore]casting with Continuous Evaluation & Composite Pipelines</b><br> Build, train and deploy pipeines with your favourite modeling library as Fold is from the ground-up extensible and lightweight.
    <br/>
    <a href="https://dream-faster.github.io/fold/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

<!-- INTRO -->

![Fold's main features](docs/images/main_features.png)

- Composite Pipelines with Continuous Validation at speed - [what does it mean for me?](#benefits-for-your-Forecasting)
- Distributed computing - [why is this importan?](#benefits-for-your-Forecasting)
- Update deployed models - [why is this importan?](#benefits-for-your-Forecasting)

<br/>

![Fold works with many third party libraries](docs/images/third_party.png)

<!-- GETTING STARTED -->
<br/>

## Installation

- Prerequisites: `python >= 3.7` and `pip`

- Install from git directly:
  ```
  pip install https://github.com/dream-faster/fold/archive/main.zip
  ```

<br/>

## Quickstart

You can quickly train your chosen models and get predictions by running:

```python
from fold.loop import train, backtest
X
y = X.squeeze()

splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
transformations = [
  DummyRegressor(strategy="constant", constant=0),
  OnlyPredictions(),
]
transformations_over_time = train(transformations, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)
```

<!-- GETTING STARTED -->

## Examples and Walkthroughs

| Link                                                                            | Dataset Type |
| ------------------------------------------------------------------------------- | ------------ |
| [⚡️ Energy Demand Walkthrough](github.com/dream-faster/fold/examples/energy)   | Energy       |
| [🚋 Basic end-to-end walkthrough](github.com/dream-faster/fold/examples/energy) | Energy       |

<br/>

## Core Features

- Supports both Regression and Classification tasks.
- Online and Mini-batch learning.
- Feature selection and other transformations on an expanding/rolling window basis
- Use any scikit-learn/tabular model natively!
- Use any univariate or sequence models (wrappers provided in [fold-models](https://github.com/dream-faster/fold-models)).
- Use any Deep Learning Time Series models (wrappers provided in [fold-models](https://github.com/dream-faster/fold-models)).
- Super easy syntax!
- Probabilistic foreacasts (currently, for Classification, soon for Regression as well).
- Hyperparemeter optimization / Model selection.

## Dream Faster Forecasting Suite

[![Dream Faster's Suite - Krisi](docs/images/dream_faster_suite_krisi.png)](https://github.com/dream-faster/krisi)
[![Dream Faster's Suite - Fold](docs/images/dream_faster_suite_fold.png)](https://github.com/dream-faster/fold)
[![Dream Faster's Suite - Fold/Models](docs/images/dream_faster_suite_fold_models.png)](https://github.com/dream-faster/fold-models)

## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

The project uses `isort` and `black` for formatting.

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.

## Benefits for your Forecasting

## Fold solves the following problems:

- Accidentally using information that wouldn't otherwise be available at the time of training/evaluation (lookahead bias). <p style="color:orange;">**→ fold allows you to use any model, transformation or function in a pipeline, while making sure it'll never have access to "future data".**<p>

- Time series Cross-validation is painful OR really slow with the existing libraries. People end up using a single train-test split when evaluating time series models, which is sub-optimal from many perspective. [Why use cross validation?](continuous-validation.md)<p style="color:orange;">
  **→ fold allows to simulate and evaluate your models like they would have performed, in reality/when deployed. Choose between sliding or expanding window.**</p>

<details>
  <summary>👇 Expand for further differentiators </summary>
<br>

- Model selection, Feature selection and Hyperparameter optimization is done on the whole time series, introducing major lookahead bias, creating unrealistic expectations of performance.<p style="color:orange;">
  **→ Allowing methodologically “correct” way to do Model selection, Feature selection and Hyperparameter Optimization (we call this the pre-validation step, done on the first window's train split).**</p>

- Too many dependencies and an either you use-all-or-none-of-it approach<p style="color:orange;">
  **→ Fold has very few hard dependencies (only pandas, numpy, tqdm and scikit-learn), and has a fraction of the number of lines of code as other major Time series libraries.**</p>

- Choosing between a time series library that only support certain kind of models.<p style="color:orange;">
  **→ Don't need to choose between `xgboost`, `sktime`, `darts` or `statsforecast` models. We will or already support them all, either natively or through [`fold-models`](https://github.com/dream-faster/fold-models)**</p>

- Most time series ML libraries have little or no support distributed computing.<p style="color:orange;">
  **→ Fold was built with distributed computing in mind. Your pipeline is automatically parallelized wherever it can be (for some extent, the rest is coming really soon)**</p>

- Using a single model, instead of ensembling, stacking or creating hybrid models.<p style="color:orange;">
  **→ Fold is _really_ flexible in what kind of pipelines you create and has native support for ensembling, stacking, hybrid models and meta-labeling. Why? [works really well for time series](https://linkinghub.elsevier.com/retrieve/pii/S0169207022001480).**</p>

- Hard to deploy models, that can't be updated.<p style="color:orange;">
  **→ Don't stop at training models, with `fold`, you can deploy with a couple of lines of code, and also update your models as new data comes in. Don't assume your models will not get out-of-date.**</p>

- We can't compare, ensemble or use online and mini-batch learning models together.<p style="color:orange;">
**→ `fold` supports both types of models natively.**</p>
</details>
</li>

## Limitations

- No intermittent time series support, very limited support for non-continuous time series.
- No multi-step ahead forecasts. If you want to forecast multiple steps ahead, transform `y` to aggregate the change over the forecasting horizon you're interested in.
- No hierarchical time series support.
