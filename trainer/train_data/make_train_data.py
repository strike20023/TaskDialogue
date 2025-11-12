# %%
import pandas as pd

def get_multiwoz_data():
    df = pd.read_csv("trainer/train_data/multiwoz.id", header=None)
    df.columns = ["task_id"]
    print(df)
    df.to_parquet("trainer/train_data/multiwoz.parquet", index=False)

    _df = pd.read_csv("data/multiwoz/testListFile.json", header=None)
    _df.columns = ["task_id"]
    df = df[df["task_id"].apply(lambda x:x in _df["task_id"].tolist())][:4]
    print(df)
    df.to_parquet("trainer/train_data/multiwoz_test.parquet", index=False)
    return df
get_multiwoz_data()
# %%
import datasets
parquet_file = "trainer/train_data/multiwoz.parquet"
datasets.load_dataset("parquet", data_files=parquet_file)["train"]
# %%
