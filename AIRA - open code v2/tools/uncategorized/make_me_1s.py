import pandas as pd
import os
from glob import glob

# Папка, где лежат все CSV файлы из распакованных архивов
INPUT_FOLDER = "BTCUSDT_Monthly_2025"
# Папка для сохранения финальных свечей
OUTPUT_FOLDER = "freqtrade_1s_data"
# Пара (обнови, если нужна другая)
PAIR = "BTCUSDT"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_file(input_path, output_path):
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('timestamp', inplace=True)

    ohlcv = df['price'].resample('1S').ohlc()
    volume = df['qty'].resample('1S').sum()

    result = ohlcv.copy()
    result['volume'] = volume
    result.dropna(inplace=True)
    result.reset_index(inplace=True)
    result['timestamp'] = (result['timestamp'].astype('int64') // 1_000_000_000)

    result[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_csv(output_path, index=False)
    print(f"✔ Converted: {os.path.basename(input_path)}")

# Поиск всех CSV файлов
all_files = glob(os.path.join(INPUT_FOLDER, "**", "*.csv"), recursive=True)

for file in sorted(all_files):
    date_part = os.path.basename(file).split('-')[-1].replace('.csv', '')
    output_file = os.path.join(OUTPUT_FOLDER, f"{PAIR}-1s-{date_part}.csv")
    convert_file(file, output_file)

print("✅ Все файлы обработаны.")
