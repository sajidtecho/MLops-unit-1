import pandas as pd

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Print basic statistics
print("Dataset Shape:", data.shape)
print("\nColumn Names:")
print(data.columns)

print("\nBasic Statistics:")
print(data.describe())