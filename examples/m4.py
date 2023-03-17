# %%
import pandas as pd

# %%
# Get the Data
df = pd.read_parquet("https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet")
df.head()

# %%
uids = df["unique_id"].unique()[:10]  # Select 10 ids to make the example faster

df = df.query("unique_id in @uids")

df = df.groupby("unique_id").tail(
    7 * 24
)  # Select last 7 days of data to make example faster

# %%
df.groupby("unique_id").plot(subplots=True)
# %matplotlib inline

# %%
# %matplotlib inline
# sns.lineplot(x="ds", y="y", data=df.groupby('unique_id'))
# %%
df.pivot(*df)
