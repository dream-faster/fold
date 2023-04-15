# Fold - Core Walkthrough

[:material-download:  Download](core_walkthrough.ipynb){ .md-button }   [:simple-googlecolab:  Open In Colab](https://colab.research.google.com/github/https://colab.research.google.com/drive/1CVhxOmbHO9PvsdHfGvR91ilJUqEnUuy8?usp=sharing){ .md-button .md-button--primary }

![Walkthrough Cover.png](https://lh3.googleusercontent.com/drive-viewer/AAOQEOT4w3Cu4i_TPzDV4WAHd3DkRuz7-4rPsre2jX05y_oanG19aCCMmi_oglzAKdRGZ-qYYTUdSoJJxE7KSc_wNkCT3GAsCA=s1600)


**Welcome 👋**

In this notebook we'll demonstrate `fold`'s powerful interface for creating, training, and cross-validating (or backtesting, if you prefer) simple and *composite* models/pipelines.

We will use the dataset from an [Energy residual load forcasting challenge](https://www.kaggle.com/competitions/energy-forecasting-data-challenge) hosted on Kaggle.

---

**By the end you will know how to:**
- Create a simple and ensemble model (composite model)
- Train multiple models / pipelines over time
- Analyze the model's simulated past performance

---

Let's start by installing:
- [`fold`](https://github.com/dream-faster/fold)
- [`fold-models`](https://github.com/dream-faster/fold-models): optional, this will be required later for third party models. Wraps eg. `XGBoost` or `StatsForecast` models to be used with `fold`.
- [`krisi`](https://github.com/dream-faster/krisi), optional. Dream Faster's Time-Series evaluation library to quickly get results.


## Installing libraries


```python
%%capture
pip install --quiet https://github.com/dream-faster/fold/archive/main.zip https://github.com/dream-faster/fold-models/archive/main.zip git+https://github.com/dream-faster/krisi.git@main matplotlib seaborn xgboost plotly prophet statsforecast statsmodels ray kaleido
```

## Data Loading and Exploration

Let's load in the data and do minimal exploration of the structure of the data. 

`fold` has a useful utility function that loads example data from our [`datasets`](https://github.com/dream-faster/datasets) GitHub repo. 

*   We are forecasting `residual_load`‡. 
*   We will shorten the dataset to `4000` rows so we have a speedier demonstration.


---

‡ *The difference between the `load` in the network and the `P` that the industrial complex is producing.*


```python
from fold.utils.dataset import get_preprocessed_dataset
from statsmodels.graphics.tsaplots import plot_acf
from krisi import score, compare
from krisi.report import plot_y_predictions
import plotly.io as pio
pio.renderers.default = "png"

X, y = get_preprocessed_dataset(
    "energy/industrial_pv_load",
    target_col="residual_load", 
    resample="H",
    deduplication_strategy="first",
    shorten=4000,
)
no_of_observation_per_day = 24
no_of_observation_per_week = no_of_observation_per_day * 7

y.plot(figsize = (20,5), grid=True);
```


    
![png](core_walkthrough_files/core_walkthrough_8_0.png)
    


The data format may be very familiar - it looks like the standard scikit-learn data.

`X` represents exogenous variables in `fold`, where a single row corresponds to a single target value. That means we currently only support univariate time-series (with exogenous variables), but soon we're extending that.

It's important that the data should be sorted and its integrity (no missing values, no duplicate indicies) should be checked before passing data to `fold`.


```python
X.head()
```





  <div id="df-b5485d31-d3be-4181-903e-dd7aa42c041a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P</th>
      <th>Gb(i)</th>
      <th>Gd(i)</th>
      <th>H_sun</th>
      <th>T2m</th>
      <th>WS10m</th>
      <th>load</th>
      <th>residual_load</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00:00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.44</td>
      <td>5.54</td>
      <td>120.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>2018-01-01 01:00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.56</td>
      <td>5.43</td>
      <td>115.5</td>
      <td>115.5</td>
    </tr>
    <tr>
      <th>2018-01-01 02:00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.04</td>
      <td>5.33</td>
      <td>120.5</td>
      <td>120.5</td>
    </tr>
    <tr>
      <th>2018-01-01 03:00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.48</td>
      <td>5.67</td>
      <td>123.5</td>
      <td>123.5</td>
    </tr>
    <tr>
      <th>2018-01-01 04:00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.95</td>
      <td>5.79</td>
      <td>136.5</td>
      <td>136.5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b5485d31-d3be-4181-903e-dd7aa42c041a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b5485d31-d3be-4181-903e-dd7aa42c041a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b5485d31-d3be-4181-903e-dd7aa42c041a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




(We'll ignore the exogenous variables until a bit later)


```python
y.head()
```




    datetime
    2018-01-01 00:00:00    115.5
    2018-01-01 01:00:00    120.5
    2018-01-01 02:00:00    123.5
    2018-01-01 03:00:00    136.5
    2018-01-01 04:00:00    138.0
    Freq: H, Name: residual_load, dtype: float64



You can see that `y` (our target) contains the next value of `X`'s "residual_load" column. 

## Time Series Cross Validation with a univariate forecaster
---

### 1. Model Building

`fold` has three core type of building blocks which you can build arbitrary sophisticated pipelines from:
- **Transformations** (classes that change, augment the data. eg: `AddHolidayFeatures` adds a column feature of holidays/weekends to your exogenous variables)
- **Models** (eg.: Sklearn, Baseline Models, third-party adapters from [`fold-models`](https://github.com/dream-faster/fold-models), like Statsmodels)
- **Composites** (eg.: `Ensemble` - takes the mean of the output of arbitrary number of 'parallel' models or pipelines)

Let's use Facebook's popular [`Prophet`](https://facebook.github.io/prophet/) library, and create in instance.

If [`fold-models`](https://github.com/dream-faster/fold-models) is installed, `fold` can take this instance without any additional wrapper class.


```python
from prophet import Prophet
prophet = Prophet()
```

### 2. Creating a Splitter

A splitter allows us to do Time Series Cross-Validation with various strategies.

`fold` supports three types of `Splitters`:
![Splitter](https://lh3.googleusercontent.com/drive-viewer/AAOQEOSAR7ICe29LSRo8umKyNIC-2c32LLdo46vSB30bwiJbYGMwFtc22rEtyWy62Eu7A0yDLimaEXOjgXx-4_PS92sqMgb0ww=s1600)


```python
from fold.splitters import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(
    initial_train_window=no_of_observation_per_week * 6,
    step=no_of_observation_per_week
)
```

Here, `initial_train_window` defines the first window size, `step` is the size of the window between folds.

We're gonna be using the first 6 weeks as our initial window, and re-train (or update, in another training mode) it every week after. We'll have 18 models, each predicting the next week's target variable.

You can also use percentages to define both, for example, `0.1` would be equivalent to `10%` of the availabel data.

### 3. Training a (univariate) Model

We could use [ray](https://www.ray.io/) to parallelize the training of multiple folds, halving the time it takes for every CPU core we have available (or deploying it to a cluster, if needed).

We pass in `None` as `X`, to indicate that we want to train a univariate model, without any exogenous variables.


```python
from fold import train_evaluate, Backend
import ray
ray.init(ignore_reinit_error=True)

scorecard, predictions, trained_pipeline = train_evaluate(prophet, None, y, splitter, backend=Backend.ray, krisi_args={"model_name":"prophet"})
```

### 4. Evaluating the results


```python
scorecard.print('minimal')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     Mean Absolute Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">95.076</span>         
          Mean Absolute Percentage Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7.9555e+13</span>     
Symmetric Mean Absolute Percentage Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.5092</span>         
                      Mean Squared Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.4779e+04</span>     
                 Root Mean Squared Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">121.57</span>         
                               R-squared - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.42684</span>        
                   Mean of the Residuals - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6.3616</span>         
     Standard Deviation of the Residuals - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">121.42</span>         
</pre>




```python
plot_y_predictions(y[predictions.index], predictions, mode="overlap", y_name="residual_load")
```


    
![png](core_walkthrough_files/core_walkthrough_27_0.png)
    


Finally, let's save the scorecard into a list, so we can compare the results later.


```python
results = [(scorecard, predictions)]
```

## Using an Ensemble (Composite) model
---

Here we will build an `Ensemble` model that leverages the output of multiple models. 

![Ensembling Models.png](https://lh3.googleusercontent.com/drive-viewer/AAOQEORyLi4ZPadHho7_C_IMdxDHxoZOt7T-y-7vMmTJ4BTubYk_4xu6hntPuK3nY1HmS4GC3DDQCKWgyqKQijheEhclhz_qYw=s1600)

### 1. Model Building with `fold-models`

We are going to define three different pipelines, each leveraging a different model and different features.



We can leverage the most popular modelling libraries, like StatsForecast, Sktime, XGBoost, etc. (the list can be found [here](https://github.com/dream-faster/fold-models)).

Let's train a [MSTL](https://arxiv.org/abs/2107.13462) model that's implemented in [StatsForecast](https://nixtla.github.io/statsforecast/models.html), that can capture multiple seasonalities, with the `WrapStatsForecast` class from `fold-models`. This is not strictly necessary, though, as the automatic wrapping also works for StatsForecast instaces as well.


```python
from statsforecast.models import MSTL
from fold_wrapper import WrapStatsForecast, WrapStatsModels

mstl = WrapStatsForecast.from_model(MSTL([24, 168]))
```

### 2. Ensembling with `fold`

Finally, let's `ensemble` the two pipelines.


```python
from fold.composites import Ensemble

univariate_ensemble = Ensemble([prophet, mstl])
```

### 3. Training all pipelines seperately and within an `ensemble`

We'll use the same `ExpandingWindowSplitter` we have defined above, to make performance comparable.


```python
from fold import train_evaluate

for name, pipeline in [
    ("mstl", mstl),
    ("univariate_ensemble",univariate_ensemble)
]:
    scorecard, predictions, pipeline_trained = train_evaluate(pipeline, None, y, splitter, krisi_args={"model_name":name})
    results.append((scorecard, predictions))
```


```python
compare([scorecard for scorecard, predictions in results])
```

                        model_name    [1mrmse           [0m 
                           prophet    [1m121.57         [0m 
                              mstl    [1m118.7          [0m 
               univariate_ensemble    [1m105.96         [0m 


We see that our Ensemble model has beaten all individual models' performance - which is very usual in the time series context.

## Using a single-step ahead forecaster (a baseline)

So far we've used models that were costly to update (or re-train) every day, therefore we were limited to training once for every week, then predicting the next week's target.

What if we could use a lightweight, "online" model, that can be updated on every timestamp?

And.. what if we just repeat the last value?

That'd be the `Naive` model you can load from `fold_wrapper`.


```python
from fold import train_evaluate
from fold_wrapper import Naive

scorecard, predictions, trained_pipeline = train_evaluate(Naive(), None, y, splitter, krisi_args={"model_name":"naive"})
results.append((scorecard, predictions))
scorecard.print("minimal")
```


      0%|          | 0/18 [00:00<?, ?it/s]



      0%|          | 0/18 [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     Mean Absolute Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">39.111</span>         
          Mean Absolute Percentage Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.1224e+14</span>     
Symmetric Mean Absolute Percentage Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2038</span>         
                      Mean Squared Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4012.9</span>         
                 Root Mean Squared Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">63.348</span>         
                               R-squared - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.84437</span>        
                   Mean of the Residuals - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.043877</span>      
     Standard Deviation of the Residuals - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">63.358</span>         
</pre>



**We call this [Continous Validation](https://dream-faster.github.io/fold/concepts/continuous-validation/).**

It looks like having access to last value really makes a difference: the baseline model beats all long-term forecasting models by a large margin.

**It's extremely important to define our forecasting task well**:
1. We need to think about what time horizon can and should forecast
2. And how frequently can we update our models.

Long-horizon (in this case, a week ahead) forecasts can be very unreliable, on the other hand, frequent, short-term forecasts are where Machine Learning shines (as we'll see in the next section).

## Using exogenous variables with Tabular Models
---

So far we have been training univariate models, and ignored all the additional, exogenous variables that come with our data.

Let's try whether using this data boost our model's performance!


### Building Models separately


We'll be using scikit-learn's `HistGradientBoostingRegressor`, their competing implementation of Gradient Boosted Trees. You don't need to wrap `scikit-learn` models or transformations when using it in `fold`, just pass it in directly to any pipeline.


```python
from sklearn.ensemble import HistGradientBoostingRegressor

tree_model = HistGradientBoostingRegressor(max_depth=10)
```




Let's add both holiday and date-time features to our previous ensemble pipeline.

The data was gathered in the Region of Hessen, Germany -- so we pass in `DE` (we can pass in multiple regions). This transformation adds another column for holidays to our `exogenous` (`X`) features.

We're also adding the current hour, and day of week as integers to our exogenous features. This is one of the ways for our tabular model to capture seasonality.



```python
from fold.transformations import AddHolidayFeatures, AddDateTimeFeatures

datetime = AddDateTimeFeatures(['hour', 'day_of_week', 'day_of_year'])
holidays = AddHolidayFeatures(['DE'])
```


Let's add a couple of lagged, exogenous values for our model. `AddLagsX` receives a tuple of column name and integer or list of lags, for each of which it will create a column in `X`.

We can easily create transformations of existing features on a rolling window basis with `AddWindowFeatures` as well, in this case, the last day's average value for all of our exogenous features.

We can "tie in" two separate pipelines with `Concat`, which concatenates all columns from all sources.


```python
from fold.transformations import AddWindowFeatures, AddLagsX
from fold.composites import Concat

tree = [
    Concat([
        AddLagsX(("all",range(1,3))),
        AddWindowFeatures([("all", 24, "mean")]),
    ]),
    datetime,
    holidays,
    tree_model
]
```

Let's see how this performs!

We can also use fold's `train`, `backtest` to decouple these functionalities.


```python
from fold import train, backtest

trained_pipeline = train(tree, X, y, splitter)
predictions = backtest(trained_pipeline, X, y, splitter)
scorecard = score(y[predictions.index], predictions.squeeze(), model_name="tabular_tree")

results.append((scorecard, predictions))
scorecard.print("minimal")
```


      0%|          | 0/18 [00:00<?, ?it/s]



      0%|          | 0/18 [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                     Mean Absolute Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">29.653</span>         
          Mean Absolute Percentage Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.1232e+14</span>     
Symmetric Mean Absolute Percentage Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.15915</span>        
                      Mean Squared Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2637.6</span>         
                 Root Mean Squared Error - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">51.357</span>         
                               R-squared - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.89771</span>        
                   Mean of the Residuals - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1.7471</span>        
     Standard Deviation of the Residuals - <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">51.336</span>         
</pre>



### Creating an Ensemble of Tabular models

First let's creat two more models:
* an Sklearn LinearRegressor
* and an XGBoostRegressor instance

We are also going to use the HistGradientBoostingRegressor pipeline that we defined prior.



```python
from sklearn.linear_model import LinearRegression

lregression = [
    AddLagsX(('all',range(1,3))),
    datetime,
    LinearRegression()
]
```


```python
from xgboost import XGBRegressor
from fold_wrapper.xgboost import WrapXGB

xgboost = [
    AddLagsX(('all',range(1,3))),
    datetime,
    WrapXGB.from_model(XGBRegressor())
]
```


```python
tabular_ensemble = Ensemble([lregression, xgboost, tree])
```


```python
scorecard, predictions, pipeline_trained = train_evaluate(tabular_ensemble, X, y, splitter, krisi_args={"model_name":"tabular_ensemble"})
results.append((scorecard, predictions))
```


      0%|          | 0/18 [00:00<?, ?it/s]



      0%|          | 0/18 [00:00<?, ?it/s]


## Comparing & Vizualising the results
---


```python
compare([scorecard for scorecard, _ in results])
```

                        model_name    [1mrmse           [0m 
                           prophet    [1m121.57         [0m 
                              mstl    [1m118.7          [0m 
               univariate_ensemble    [1m105.96         [0m 
                             naive    [1m63.348         [0m 
                      tabular_tree    [1m51.357         [0m 
                  tabular_ensemble    [1m48.229         [0m 


In this simplistic, unfair comparison, it looks like the tabular models (and the Naive baseline) that have access to the previous value (and the exogenous variables) outperform the univariate models that are only re-trained every week. 

We can't really draw general conclusions from this work, though. 

Unlike NLP and Computer vision, Time Series data is very heterogeneous, and a Machine Learning approach that works well for one series may be an inferior choice for your specific usecase.

---

But now we have an easy way to compare the different pipelines, with unprecedented speed, by using a unified interface, with [fold](https://github.com/dream-faster/fold). 





```python
all_predictions =[predictions.squeeze().rename(scorecard.metadata.model_name) for scorecard, predictions in results]

plot_y_predictions(y[predictions.index], all_predictions, y_name="residual_load", mode='seperate')
```


    
![png](core_walkthrough_files/core_walkthrough_63_0.png)
    


Want to know more?
Visit [fold's Examples page](https://dream-faster.github.io/fold/), and access all the necessary snippets you need for you to build a Time Series ML pipeline!
