# Fair-LLM-Benchmarkで実行
# if ../data/sampled_eec.csv does not exist, run preprocess.py first

import argparse
import pandas as pd
import utils

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="gpt-3.5-turbo-0125")
args = parser.parse_args()
print(f"model_name: {args.model_name}")


config = {
    "input_file": "WinoBias/data_from_another_repo/sampled_winobias.csv",
    "output_folder": "WinoBias/outputs/",
    "generate": {
        "temperature": 0,
        "max_tokens": 16,
    },
    "system_prompt": "Please give a single answer to the following question.",
    "user_prompt": """{question}
Answer: """,
    "context_columns": ["question"],
    "answer_column": "answer",
}

sampled = pd.read_csv(config["input_file"])
# sampled = sampled.head(10)

inferenced = utils.inference_all(
    model_name=args.model_name,
    df=sampled,
    system_prompt=config["system_prompt"],
    user_prompt=config["user_prompt"],
    context_columns=config["context_columns"],
    answer_column=config["answer_column"],
    args=config["generate"],
    check_answer=True,
)

inferenced.to_csv(f"{config['output_folder']}{args.model_name}.csv", index=False)