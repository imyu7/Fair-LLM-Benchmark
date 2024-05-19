# Fair-LLM-Benchmarkで実行
# This script samples 2 examples for each combination of template
# ../data/sampled_eec.csv will be created
# if it already exists, you don't need to run this script

import pandas as pd

config = {
    "input_file": "Equity-Evaluation-Corpus/original/Equity-Evaluation-Corpus.csv",
    "output_file": "Equity-Evaluation-Corpus/data/sampled_eec.csv",
}

eec = pd.read_csv(config["input_file"])

Races = ["African-American", "European"]
Genders = ["male", "female"]
Emotions = ["anger", "fear", "joy", "sadness"]
Templates = eec.Template.value_counts().index.tolist()

sampled_eec = pd.DataFrame(columns=eec.columns)

for template in Templates[:4]:
    for gender in Genders:
        for race in Races:
            for emotion in Emotions:
                row = eec.loc[(eec["Template"] == template) & (eec["Gender"] == gender) & (eec["Race"] == race) & (eec["Emotion"] == emotion)]
                sampled_eec = pd.concat([sampled_eec, row.sample(2, random_state=42)])

sampled_eec.reset_index(drop=True, inplace=True)
sampled_eec.to_csv(config["output_file"], index=False)