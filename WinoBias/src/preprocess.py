# Fair-LLM-Benchmarkで実行
# ../data/sampled_winobias.csv will be created
# if it already exists, you don't need to run this script

import pandas as pd
import json
import random

random.seed(42)

config = {
    "input": {
        "folder": "WinoBias/data_from_another_repo/",
        "files": ["pro_stereotyped_type1.json", "anti_stereotyped_type1.json"],
    },
    "output_file": "WinoBias/data_from_another_repo/sampled_winobias.csv",
}


with open(config["input"]["folder"] + config["input"]["files"][0]) as f:
  pro_1 = json.load(f)

with open(config["input"]["folder"] + config["input"]["files"][1]) as f:
  anti_1 = json.load(f)


df_pro_1 = pd.DataFrame(pro_1['testset'])
df_pro_1['bias_type'] = "pro"
df_pro_1['scentence_type'] = 1

df_anti_1 = pd.DataFrame(anti_1['testset'])
df_anti_1['bias_type'] = "anti"
df_anti_1['scentence_type'] = 1

chosen_index = random.sample(range(380), 100)
df_pro_1, df_anti_1 = df_pro_1.iloc[chosen_index], df_anti_1.iloc[chosen_index]

df = pd.concat([df_pro_1, df_anti_1]).reset_index(drop=False)


print(df.shape)

# save
df.to_csv(config["output_file"], index=False)