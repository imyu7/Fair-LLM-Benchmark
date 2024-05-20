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
df["stereotyped_groups"] = df["additional_metadata"].apply(lambda x: x["stereotyped_groups"])

for i in range(3):
    df[f"ans{i}_info"] = df["answer_info"].apply(lambda x: x[f"ans{i}"][1])

for index, row in df.iterrows():
    if row["ans0_info"] == "unknown": 
        df.at[index, "unknown_choice"] = '0'
    elif row["ans1_info"] == "unknown":
        df.at[index, "unknown_choice"] = '1'
    elif row["ans2_info"] == "unknown":
        df.at[index, "unknown_choice"] = '2'
    else:
        df.at[index, "unknown_choice"] = '-1'
    if row["ans0_info"] in row["stereotyped_groups"]:
        df.at[index, "stereotyped_choice"] = '0'
    elif row["ans1_info"] in row["stereotyped_groups"]:
        df.at[index, "stereotyped_choice"] = '1'
    elif row["ans2_info"] in row["stereotyped_groups"]:
        df.at[index, "stereotyped_choice"] = '2'
    else:
        df.at[index, "stereotyped_choice"] = '-1'
    
for i in range(3):
    df[f"ans{i}_info"] = df[f"ans{i}_info"].apply(lambda x: "F" if x == "woman" or x == "girl" else "M" if x == "man" or x == "boy" else x)

def find_unknown_choice(row):
    for i in range(3):
        if row[f"ans{i}_info"] == "unknown":
            return str(i)
    return '-1'

def find_stereotyped_choice(row):
    for i in range(3):
        if row[f"ans{i}_info"] in row["stereotyped_groups"]:
            return str(i)
    return '-1'


df.drop(columns=["answer_info", "additional_metadata", "stereotyped_groups", "ans0_info", "ans1_info", "ans2_info"], inplace=True)
print(df.shape)
df.to_csv(config["output_file"], index=False)