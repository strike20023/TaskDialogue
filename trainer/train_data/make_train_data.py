# %%
import pandas as pd

def get_multiwoz_data():
    df = pd.read_csv("trainer/train_data/multiwoz.id", header=None)
    df.columns = ["task_id"]
    df.to_parquet("trainer/train_data/multiwoz.parquet", index=False)
    return df
get_multiwoz_data()
# %%