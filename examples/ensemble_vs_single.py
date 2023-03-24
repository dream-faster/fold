# %% [markdown]
## The Power of Ensembling
# ---
### Energy Nowcasting Walkthrough
# In this notebook we'll create three simple pipelines: two with different settings of ARIMA and a composite model that combines the two.
# We then explore how the models perform individually and when ensembled.
# All of the pipeline are tested on the same data: Let's forecast the residual load in the network of the Bundesland of Hessen.

from fold_models.statsforecast import WrapStatsForecast
from krisi import compare, score
from statsforecast.models import ARIMA

# %%
from utils import load_dataset

from fold.composites.ensemble import Ensemble
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter

# %% [markdown]
# Let's load in the data.
# 1. Some indexes are duplicate (due to summer-time/winter-time change).
#   We deduplicate those rows and only keep the first instance of each duplicate.
# 2. Then we take out our target column (`y`) and pop it from our Exogenous DataFrame (`X`)

# %%
target = "residual_load"
X = load_dataset("energy/industrial_pv_load").set_index("time", drop=True)
X = X[~X.index.duplicated(keep="first")]
y = X.pop(target)


# %% [markdown]
# Let's first define a Splitter that allow us to run the same Continous Validation on all three pipelines
# We also define a `scorecards` list where we collect the results of all three evalutations.

# %%
splitter = ExpandingWindowSplitter(initial_train_window=0.15, step=0.1)
scorecards = []

# %% [markdown]
### Pipeline 1 - `ARIMA` with order `(1, 1, 0)`
# Let's create a simple ARIMA model with Statforecast wrapped with fold-models

# %%
pipeline_arima_1 = WrapStatsForecast(ARIMA, {"order": (1, 1, 0)}, use_exogenous=False)

## Training the model
transformations_over_time = train(pipeline_arima_1, None, y, splitter)

## Evaluating the model
pred = backtest(transformations_over_time, X, y, splitter)

sc = score(y[pred.index], pred.squeeze(), "Pipeline_ARIMA_1")
sc.print_summary(extended=False)
scorecards.append(sc)

# %% [markdown]
### Pipeline 1 - `ARIMA` with order `(2, 1, 1)`
# Let's create another simple ARIMA model with Statforecast wrapped with fold-models

# %%
pipeline_arima_2 = WrapStatsForecast(ARIMA, {"order": (2, 1, 1)}, use_exogenous=False)

## Training the model
transformations_over_time = train(pipeline_arima_2, None, y, splitter)

## Evaluating the model
pred = backtest(transformations_over_time, X, y, splitter)

sc = score(y[pred.index], pred.squeeze(), "Pipeline_ARIMA_2")
sc.print_summary(extended=False)
scorecards.append(sc)

# %% [markdown]
### Composite Pipeline - Both previous ARIMAs Ensembled
# Let's create another simple ARIMA model with Statforecast wrapped with fold-models

# %%
pipeline_ensemble = Ensemble([pipeline_arima_1.clone(), pipeline_arima_2.clone()])

## Training the model
transformations_over_time = train(pipeline_ensemble, X, y, splitter)

## Evaluating the model
pred = backtest(transformations_over_time, X, y, splitter)

sc = score(y[pred.index], pred.squeeze(), "Pipeline_Ensemble")
sc.print_summary(extended=False)
scorecards.append(sc)

# %% [markdown]
### Comparing the results

# %%
compare(scorecards)
