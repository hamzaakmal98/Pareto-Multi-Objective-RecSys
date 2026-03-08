import pandas as pd

df = pd.read_csv("../data/raw/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv")

print("Shape:", df.shape)
print("Columns:")
print(df.columns)