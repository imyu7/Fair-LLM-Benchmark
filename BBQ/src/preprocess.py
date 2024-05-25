# Fair-LLM-Benchmarkで実行
# ../data/sampled_bbq.csv will be created
# if it already exists, you don't need to run this script

import pandas as pd

config = {
    "input_folder": "BBQ/data/",
    "output_file": "BBQ/data/sampled_bbq.csv",
}

# choose category
category = [
    "Age", 
    "Race_ethnicity", 
    # "Sexual_orientation", 
    # "Disability_status", 
    # "Race_x_SES", 
    "Gender_identity", 
    # "Race_x_gender", 
    # "Nationality", 
    "Religion", 
    "Physical_appearance", 
    # "SES"
]

# read data
df = pd.DataFrame()
for cat in category:
    df_cat = pd.read_json(f"{config['input_folder']}{cat}.jsonl", lines=True, orient='records')
    df_cat = df_cat.head(40) # sample 40 data
    df = pd.concat([df, df_cat], axis=0)

# preprocess
def change_gender_terms(text):
    if text == "woman" or text == "girl":
        return "F"
    elif text == "man" or text == "boy":
        return "M"
    else:
        return text

df["unknown_choice"] = df["answer_info"].apply(lambda x: 0 if x["ans0"][1] == "unknown" else 1 if x["ans1"][1] == "unknown" else 2 if x["ans2"][1] == "unknown" else '-1')
df["stereotyped_choice"] = df.apply(
    lambda x: 
    0 if change_gender_terms(x["answer_info"]["ans0"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else 1 if change_gender_terms(x["answer_info"]["ans1"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else 2 if change_gender_terms(x["answer_info"]["ans2"][1]) in x["additional_metadata"]["stereotyped_groups"] 
    else '-1', 
    axis=1)

assert len(df[df["unknown_choice"] == '-1']) == 0
assert len(df[df["stereotyped_choice"] == '-1']) == 0


df.drop(columns=["answer_info", "additional_metadata"], inplace=True)
print(df.shape)

# save
df.to_csv(config["output_file"], index=False)