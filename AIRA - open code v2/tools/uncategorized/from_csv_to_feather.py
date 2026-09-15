import pandas as pd
import glob

# Путь к CSV-файлам
csv_files = glob.glob("freqtrade_1s_data/BTCUSDT-1s.csv")

for file in csv_files:
    df = pd.read_csv(file)
    feather_file = file.replace(".csv", ".feather")
    df.to_feather(feather_file)
    print(f"Сохранено: {feather_file}")
