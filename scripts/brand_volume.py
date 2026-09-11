import pandas as pd

df = pd.read_csv("data/twcs.csv", usecols=["author_id", "inbound"])
brands = df[df["inbound"] == False]["author_id"].value_counts().head(20)
print(brands)
