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
    <img src="https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/logo.svg" alt="Logo" width="90" >
  </a>
<h3 align="center"><b>FOLD</b><br> <i>(/fold/)</i></h3>
  <p align="center">
    <b>A Time Series Continuous Evaluation (Cross-validation) library that lets you build, deploy and update composite models easily. An order of magnitude speed-up, combined with flexibility and rigour.</b><br>
    <br/>
    <a href="https://dream-faster.github.io/fold/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

<!-- INTRO -->

![Fold's main features](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/main_features.svg) 

- Composite Pipelines with Continuous Validation (Cross Validation) - [What does that mean?](#Fold-is-different)
- Distributed computing - [Why is this important?](#Fold-is-different)
- Update deployed models - [why is this importan?](#Fold-is-different)

Continuous validation prevents you from accidentally using information that wouldn't otherwise be available at the time of training/evaluation (lookahead bias).


<br/>

![Fold works with many third party libraries](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/third_party.svg)


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
import pandas as pd
from fold import train_evaluate
from fold.transformations import OnlyPredictions
from fold.models.dummy import DummyRegressor

X = pd.read_csv(
    "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets/weather/historical_hourly_la.csv",
    index_col=0,
    parse_dates=True,
)[:1000]
y = X.pop("temperature")

transformations = [
    DummyRegressor(0),
    OnlyPredictions(),
]
splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
scorecard, prediction, trained_transformations = train_evaluate(
    transformations, X, y, splitter, with_krisi = False
)  
```

You can further set `with_krisi` [Optional, Default = `False`], to True to get a full report. Requires krisi to be installed (`pip install krisi`)





## Fold is different

- Continuous Validation (Time Series cross-validation) at lightning speed.<br/>
  <span style="color:orange;">**→ fold allows to simulate and evaluate your models like they would have performed, in reality/when deployed, with clever use of paralellization and design.**</span>

- Create composite models: ensembles, hybrids, stacking pipelines, easily.<br/>
  <span style="color:orange;">**→ Underutilized, but [the easiest, fastest way to increase performance of your Time Series models.](https://linkinghub.elsevier.com/retrieve/pii/S0169207022001480)**
  </span>

- Built with Distributed Computing in mind.<br/>
  <span style="color:orange;">**→ Deploy your pipelines to a cluster with `ray`, and use `modin` to handle huge, out-of-memory datasets.**</span>

- Bridging the gap between Online and Mini-Batch learning.<br/>
  <span style="color:orange;">**→ Mix and match `xgboost` with ARIMA, in a single pipeline, update your models, preprocessing steps on every timestamp, if that's desired.**</span>

- Update your deployed models, easily, as new data flows in.<br/>
  <span style="color:orange;">**→ Real world is not static. Let your models adapt, without the need to re-train from scratch.**</span>




</li>

<!-- GETTING STARTED -->

## Examples and Walkthroughs

| Link                                                                           | Dataset Type |
| ------------------------------------------------------------------------------ | ------------ |
| [⚡️ Energy Demand Walkthrough](github.com/dream-faster/fold/examples/energy)    | Energy       |
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
- Hyperparemeter optimization / Model selection. (coming in early April!)

## Our open source Time Series toolkit

[![Krisi](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/dream_faster_suite_krisi.svg)](https://github.com/dream-faster/krisi)
[![Fold](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/dream_faster_suite_fold.svg)](https://github.com/dream-faster/fold)
[![Fold/Models](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/dream_faster_suite_fold_models.svg)](https://github.com/dream-faster/fold-models)


## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

The project uses `isort` and `black` for formatting.

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.


## Limitations

- No intermittent time series support, very limited support for non-continuous time series.
- No multi-step ahead forecasts. If you want to forecast multiple steps ahead, transform `y` to aggregate the change over the forecasting horizon you're interested in.
- No hierarchical time series support.
