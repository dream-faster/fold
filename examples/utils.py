import pandas as pd


def load_dataset(
    dataset_name: str,
    base_path: str = "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets",
) -> pd.DataFrame:
    return pd.read_csv(f"{base_path}/{dataset_name}.csv")
