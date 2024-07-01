#!/usr/bin/env python3

from glob import glob
import pandas as pd
from tqdm.auto import tqdm

files = glob("*/**/*.csv", recursive=True)
print(len(files))

results = []
for f in tqdm(files):
    result = pd.read_csv(f).length_cm.describe()
    result["filename"] = f
    results.append(result)

df = pd.DataFrame(results)
print(df)
failed = df["count"] == 0
print("Failures:")
print(failed.value_counts())
df.to_csv("results.csv")
