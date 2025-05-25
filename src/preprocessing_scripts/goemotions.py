import pandas as pd
import numpy as np


def get_emotions(row, emotion_columns):
    emotions = [emotion for emotion in emotion_columns if row[emotion] == 1]

    if len(emotions) == 1:
        return emotions[0]
    else:
        return np.nan


def preprocess_goemotions(dataset_id):
    df = pd.read_csv(f"../../data/goemotions_{dataset_id}.csv")

    df_processed = df[["text"]].copy()

    emotion_columns = df.columns[df.columns.get_loc("admiration") :]

    df_processed["label"] = df[emotion_columns].apply(lambda row: get_emotions(row, emotion_columns), axis=1)
    df_processed.dropna(inplace=True)
    df_processed.to_csv(f"../../data/goemotions_processed_{dataset_id}.csv", index=False)


if __name__ == "__main__":
    preprocess_goemotions(1)
