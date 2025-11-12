# %%
from trainer.multiwoz.rollout_agent import TrainingTask
import pandas as pd
import json
def get_multiwoz_data():
    with open("data/multiwoz/data.json", "r") as f:
        raw_data = json.load(f)
    data = []
    for data_index, data_item in raw_data.items():
        data.append(
            TrainingTask(
                dialogue_data={data_index:data_item}
            ).model_dump()
        )
    return data
d = get_multiwoz_data()
import pandas as pd
pd.DataFrame(d).to_parquet("trainer/multiwoz/train_data.parquet", engine="pyarrow")

# %%
