import pandas as pd
import os
import glob

path = os.getcwd()
csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))

for f in csv_files:
    df = pd.read_csv(f)
    filename = os.path.basename(f.split("\\")[-1])
    profit = df['Profit'].sum()
    margin = df['Margin'].sum()
    print(f"File: {filename} - Profit: {profit} - Margin: {margin}")
