<!-- # Fold -->

<p align="center" style="display:flex; width:100%; align-items:center; justify-content:center;">
  <a style="margin:2px" href="https://dream-faster.github.io/fold/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/dream-faster/fold/docs.yaml?logo=readthedocs"></a>
  <a style="margin:2px" href="https://codecov.io/gh/dream-faster/fold" ><img src="https://codecov.io/gh/dream-faster/fold/branch/main/graph/badge.svg?token=Z7I2XSF188"/></a>
  <a style="margin:2px" href="https://github.com/dream-faster/fold/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/dream-faster/fold/actions/workflows/tests.yaml/badge.svg"/></a>
  <a style="margin:2px" href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
  <a style="margin:2px" href="https://calendly.com/mark-szulyovszky/consultation"><img alt="Discord Community" src="https://shields.io/badge/-Speak%20with%20us-orange?logo=minutemailer&logoColor=white"></a>
</p>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://dream-faster.github.io/fold/">
    <img src="https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/logo.svg" alt="Logo" width="90" >
  </a>
<h3 align="center"><b>FOLD</b><br> <i>(/fold/)</i></h3>
  <p align="center">
    <b>A <a href="https://dream-faster.github.io/fold/concepts/continuous-validation/">Time Series Continuous Validation</a> library that lets you build, deploy and update Composite Models easily. An order of magnitude speed-up, combined with flexibility and rigour.</b><br>
    <br/>
    <a href="https://dream-faster.github.io/fold/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

<!-- INTRO -->

![Fold's main features](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/main_features.svg)

- Composite Models with Continuous Validation - [What does that mean?](https://dream-faster.github.io/fold/concepts/continuous-validation/)
- Distributed computing - [Why is this important?](#Fold-is-different)
- Update deployed models (coming in May) - [Why is this important?](#Fold-is-different)

![Fold works with many third party libraries](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/third_party.svg)

<!-- GETTING STARTED -->

## Installation

- Prerequisites: `python >= 3.7` and `pip`

- Install from git directly:
  ```
  pip install https://github.com/dream-faster/fold/archive/main.zip
  ```

## Quickstart

You can quickly train your chosen models and get predictions by running:

```py
from sklearn.ensemble import RandomForestRegressor
from statsforecast.models import ARIMA
from fold import ExpandingWindowSplitter, train_evaluate
from fold.composites import Ensemble
from fold.transformations import OnlyPredictions
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la", target_col="temperature", shorten=1000
)

pipeline = [
    Ensemble(
        [
            RandomForestRegressor(),
            ARIMA(order=(1, 1, 0)),
        ]
    ),
    OnlyPredictions(),
]
splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2)
scorecard, prediction, trained_pipelines = train_evaluate(pipeline, X, y, splitter)
```

Thinking of using `fold`? We'd love to hear about your use case and help, [please book a free 30-min call with us](https://calendly.com/mark-szulyovszky/consultation)!

(If you install `krisi` by running `pip install krisi` you get an extended report back, rather than a single metric.)

## Fold is different

- Time Series Continuous Validation at lightning speed.<br/>
  <span style="color:orange;">**→ fold allows to simulate and evaluate your models like they would have performed, in reality/when deployed, with clever use of paralellization and design.**</span>

- Create composite models: ensembles, hybrids, stacking pipelines, easily.<br/>
  <span style="color:orange;">**→ Underutilized, but [the easiest, fastest way to increase performance of your Time Series models.](https://linkinghub.elsevier.com/retrieve/pii/S0169207022001480)**
  </span>

- Built with Distributed Computing in mind.<br/>
  <span style="color:orange;">**→ Deploy your research and development pipelines to a cluster with `ray`, and use `modin` to handle out-of-memory datasets (full support for modin is coming in April).**</span>

- Bridging the gap between Online and Mini-Batch learning.<br/>
  <span style="color:orange;">**→ Mix and match `xgboost` with ARIMA, in a single pipeline. Boost your model's accuracy by updating them on every timestamp, if desired.**</span>

- Update your deployed models, easily, as new data flows in.<br/>
  <span style="color:orange;">**→ Real world is not static. Let your models adapt, without the need to re-train from scratch.**</span>

<!-- GETTING STARTED -->

## Examples and Walkthroughs

<table style="width:100%">
  <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Dataset Type</th>
    <th>Docs Link</th>
    <th>Colab</th>
  </tr>
  <tr>
    <td> 
     ⚡️ Core Walkthrough
    </td>
    <td>Walkthrough</td>
    <td>Energy</td>
    <td>  
      <a href='https://dream-faster.github.io/fold/walkthroughs/core_walkthrough/' target="_blank">Notebook</a>
    </td>
    <td>
     <a href='https://colab.research.google.com/drive/1CVhxOmbHO9PvsdHfGvR91ilJUqEnUuy8?usp=sharing' target="_blank">Colab</a>
    </td>
  </tr>
  <tr>
    <td> 
    🚄 Speed Comparison of Fold to other libraries
    </td>
    <td>Walkthrough</td>
    <td>Weather</td>
    <td> 
        <a href='https://dream-faster.github.io/fold/walkthroughs/benchmarking_sktime_fold/' target="_blank">
        Notebook
        </a>
    </td>
    <td>
        <a href='https://colab.research.google.com/drive/1iLXpty-j1kpDCzLM4fCsP3fLoS_DFN1C?usp=sharing' target="_blank"> 
        Colab
        </a>
    </td>
  </tr>
  <tr>
    <td> 
    📚 Example Collection
    </td>
    <td>Example</td>
    <td>Weather & Synthetic</td>
    <td> 
        <a href='https://dream-faster.github.io/fold/generated/gallery/' target="_blank">
        Collection Link
        </a>
    </td>
    <td>
    -
    </td>
  </tr>
</table>

<br/>

## Core Features

- Supports both Regression and Classification tasks.
- Online and Mini-batch learning.
- Feature selection and other transformations on an expanding/rolling window basis
- Use any scikit-learn/tabular model natively!
- Use any univariate or sequence models (wrappers provided in [fold-wrappers](https://github.com/dream-faster/fold-wrappers)).
- Use any Deep Learning Time Series models (wrappers provided in [fold-wrappers](https://github.com/dream-faster/fold-wrappers)).
- Super easy syntax!
- Probabilistic foreacasts (currently, for Classification, full support coming in April).
- Hyperparemeter optimization / Model selection. (coming in early April!)

## What is Continuous Validation?

![Continous Validation](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/technical_diagrams/continous_validation.svg)
It's Time Series Cross-Validation, plus:
Inside a test window, and during deployment, fold provides a way for a model to access the last value.
[Learn more](https://dream-faster.github.io/fold/concepts/continuous-validation/)

## Our Open-core Time Series Toolkit

[![Krisi](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_krisi.svg)](https://github.com/dream-faster/krisi)
[![Fold](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold.svg)](https://github.com/dream-faster/fold)
[![Fold/Models](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold_models.svg)](https://github.com/dream-faster/fold-models)
[![Fold/Wrappers](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold_wrappers.svg)](https://github.com/dream-faster/fold-wrappers)

If you want to try them out, we'd love to hear about your use case and help, [please book a free 30-min call with us](https://calendly.com/mark-szulyovszky/consultation)!

## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.

## Licence & Usage

Fold is our open-core Time Series engine. It's available under the MIT + Common Clause licence.
We want to **bring much-needed transparency, speed and rigour** to the process of building Time Series ML models. We're building multiple products with and on top of it.

It will be always free for research useage, but we will be charging for deployment, and for extra features that are results of our own resource-intensive R&D. We're building a sustainable business, that supports the ecosystem long-term.

## Limitations

- No intermittent time series support, very limited support for missing values.
- No hierarchical time series support.
