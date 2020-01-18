import pandas as pd

df = pd.read_csv("data/meli/train.csv")
df = df[df["language"]=="spanish"]
df.to_csv("data/lang_model_eval/es-train.csv", index=False)
