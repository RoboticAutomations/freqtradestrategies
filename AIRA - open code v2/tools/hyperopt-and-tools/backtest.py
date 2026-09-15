import os
import json
from datetime import timedelta, datetime
import random
import statistics

os.makedirs('results', exist_ok=True)

def generate_price_data(start_date_str, end_date_str):
    data = []
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    while start_date <= end_date:
        date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        open_price = random.uniform(20000, 50000)
        close_price = open_price + (random.random() * 10000 - 5000) / 10
        data.append({
            'date': date,
            'Open': open_price,
            'Close': close_price,
            'High': max(open_price, close_price),
            'Low': min(open_price, close_price)
        })
        start_date += timedelta(minutes=15)

    return data

def backtest_strategy(btc_data):
    btc_data['pct_drop_from_open'] = ((btc_data['Open'] - btc_data['daily_open']) / btc_data['daily_open']) * 100
    btc_data['entry_signal'] = (btc_data['pct_drop_from_open'] <= -3.0) & (btc_data['volume_threshold_met'])
    btc_data['entry_bar'] = btc_data['entry_signal'].astype(int)

    entries = []
    for i in range(1, len(btc_data)):
        if btc_data['entry_signal'][i] and not btc_data['position_active'][i]:
            entry_price = btc_data['Close'][i]
            initial_drop = btc_data['initial_drop'][i]
            recovery_target = btc_data['recovery_target'][i]

            entries.append({
                'entry_time': btc_data.index[i],
                'entry_price': entry_price,
                'initial_drop_pct': initial_drop,
                'recovery_target': recovery_target
            })

    total_return_pct = 0.0
    max_drawdown_pct = 0.0
    win_rate_pct = 0.0
    profit_factor = 0.0
    trades_count = len(entries)
    wins_count = 0

    for entry in entries:
        if entry['recovery_target'] > entry['entry_price']:
            exit_time = entry['entry_time'] + timedelta(hours=12)  # Exit after 12 hours
            exit_price = random.uniform(entry['entry_price'], entry['recovery_target'])
            profit_factor += (exit_price - entry['entry_price']) / abs(entry['entry_price'] - entry['recovery_target'])

            if exit_price > entry['recovery_target']:
                win_rate_pct += 100
                wins_count += 1

            total_return_pct += ((exit_price - entry['entry_price']) / entry['entry_price']) * 100

        else:
            max_drawdown_pct = max(max_drawdown_pct, (entry['recovery_target'] - entry['entry_price']) / entry['entry_price'])

    sharpe_ratio = statistics.stdev(trades_count) ** 2 if trades_count > 0 else float('inf')
    summary = {
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': len(entries),
        'win_rate_pct': win_rate_pct / trades_count if trades_count > 0 else 0.0,
        'profit_factor': profit_factor / trades_count if trades_count > 1 else float('inf')
    }

    with open('results/backtest_results.json', 'w') as f:
        json.dump(summary, f)

if __name__ == "__main__":
    start_date_str = "2023-01-01"
    end_date_str = "2024-01-01"

    btc_data = generate_price_data(start_date_str, end_date_str)
    backtest_strategy(btc_data)

    print("Backtest completed. Results saved in results/backtest_results.json")