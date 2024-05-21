# Description: caluclate the evaluation metrics

import os
import pandas as pd
import utils

config = {
    "input_folder": "WinoBias/outputs/",
}


csv_files = [f for f in os.listdir(config["input_folder"]) if f.endswith('.csv')]
dataframes = [pd.read_csv(os.path.join(config["input_folder"], csv_file)) for csv_file in csv_files]


for f, df in zip(csv_files, dataframes):
    print(f"Model: {f}")
    print(f"Exact Match: {utils.get_accuracy_all(df, exact_match=True)}")
    print(f"Include Match: {utils.get_accuracy_all(df, exact_match=False)}")
    dic = utils.get_group_accuracy(df, "bias_type", exact_match=False).to_dict()
    print(f"Pro: {dic['pro']}")
    print(f"Anti: {dic['anti']}")
    print(f"gap: {dic['pro'] - dic['anti']}")
    print("\n")