"""
STEP 1: Explore the dataset before doing anything else.
Why: you need to know column types, missing values, and class balance
BEFORE you pick a model or write preprocessing code. Skipping this step
is the #1 cause of silently broken pipelines.
"""
import pandas as pd

df = pd.read_csv("/home/claude/careerlens/data/cpp.csv")

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nTarget class count:", df["Career"].nunique())
print("\nMin/max samples per class:")
counts = df["Career"].value_counts()
print("min:", counts.min(), "max:", counts.max(), "mean:", counts.mean().round(1))
print("\nNumeric feature ranges:\n", df.describe().T[["min", "max", "mean"]])
