import pandas as pd
from pathlib import Path

# Locate the dataset
data_path = Path(__file__).resolve().parent.parent / "data" / "diabetic_data.csv"

# Load the healthcare dataset
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nDataset information:")
df.info()