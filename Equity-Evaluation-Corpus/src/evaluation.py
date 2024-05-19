# Description: caluclate the evaluation metrics

import os
import pandas as pd
import utils

config = {
    "input_folder": "Equity-Evaluation-Corpus/outputs/",
}


csv_files = [f for f in os.listdir(config["input_folder"]) if f.endswith('.csv')]
dataframes = [pd.read_csv(os.path.join(config["input_folder"], csv_file)) for csv_file in csv_files]

for f, df in zip(csv_files, dataframes):
    print(f"Model: {f}")
    print(f"Exact Match: {utils.get_accuracy_all(df, exact_match=True)}")
    print(f"Include Match: {utils.get_accuracy_all(df, exact_match=False)}")
    male_accuracy, female_accuracy = utils.get_accuracy_along_gender(df, exact_match=False)
    print(f"accuracy along gender (Include Match):\n\tmale: {male_accuracy}\n\tfemale: {female_accuracy}")
    african_accuracy, european_accuracy = utils.get_accuracy_along_race(df, exact_match=False)
    print(f"accuracy along race (Include Match):\n\tAfrican-American: {african_accuracy}\n\tEuropean: {european_accuracy}")
    print("\n")