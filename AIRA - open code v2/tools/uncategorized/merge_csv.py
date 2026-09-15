import pandas as pd
import os
from glob import glob

INPUT_FOLDER = "freqtrade_1s_data"
OUTPUT_FILE = "BTCUSDT-1s.csv"  # или "BTC/USDT-1s.csv" для дробного формата

# Собираем все CSV
all_files = glob(os.path.join(INPUT_FOLDER, "*.csv"))

# Читаем и объединяем
df_all = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# Удаляем дубликаты и сортируем
df_all.drop_duplicates(subset="timestamp", inplace=True)
df_all.sort_values("timestamp", inplace=True)

# Сохраняем итог
df_all.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Итоговый файл сохранён: {OUTPUT_FILE}")
