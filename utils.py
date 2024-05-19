from openai import OpenAI
import os
from dotenv import load_dotenv
import re


load_dotenv(".env")

client = OpenAI()


def get_completion(model, system_prompt, user_prompt, args):
    if model == 'gpt-3.5-turbo-0125' or model == 'gpt-4o-2024-05-13':
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **args,
        )
        return completion.choices[0].message.content
    elif model == "davinci-002" or model == "gpt-3.5-turbo-instruct-0914":
        response = client.completions.create(
            model=model,
            prompt=f"{system_prompt}\n{user_prompt}",
            **args,
        )
        return response.choices[0].text


def exact_match(prediction, answer):
    prediction = prediction.lower()
    return prediction == answer

def include_match(prediction, answer):
    prediction = prediction.lower()
    return answer in prediction

def inference_all(model, df, system_prompt, user_prompt, question_column, answer_column, args):
    df["Prediction"] = df.apply(lambda row: get_completion(model, system_prompt, user_prompt.format(sentence=row[question_column]), args), axis=1)
    df["ExactMatch"] = df.apply(lambda row: exact_match(row["Prediction"], row[answer_column]), axis=1)
    df["IncludeMatch"] = df.apply(lambda row: include_match(row["Prediction"], row[answer_column]), axis=1)
    return df

def get_accuracy_all(df, exact_match=True):
    if exact_match:
        return df["ExactMatch"].mean()
    else:
        return df["IncludeMatch"].mean()
    

# EEC用
def get_accuracy_along_gender(df, exact_match=True):
    male_df = df[df["Gender"] == "male"]
    female_df = df[df["Gender"] == "female"]
    male_accuracy = get_accuracy_all(male_df, exact_match)
    female_accuracy = get_accuracy_all(female_df, exact_match)
    return male_accuracy, female_accuracy

def get_accuracy_along_race(df, exact_match=True):
    african_american_df = df[df['Race'] == 'African-American']
    european_df = df[df['Race'] == 'European']
    african_american_accuracy = get_accuracy_all(african_american_df, exact_match)
    european_accuracy = get_accuracy_all(european_df, exact_match)
    return african_american_accuracy, european_accuracy