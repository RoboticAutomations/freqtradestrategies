#!/usr/bin/env python3

"""
Script: equity_progress_plot.py

Descrizione:
- Legge il CSV generato dalla strategia (user_data/backtest_results/equity_progress.csv)
- Costruisce un grafico con 2 curve nel tempo:
  * equity_est (asse sinistro)
  * used_pct (asse destro, in percentuale)
- Salva il grafico su file PNG ed opzionalmente lo mostra a schermo.

Uso:
  python plot/equity_progress_plot.py \
    --csv user_data/backtest_results/equity_progress.csv \
    --out plot/equity_progress.png \
    --show
"""

import argparse
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt


def read_equity_csv(csv_path: str):
    times = []
    equity = []
    used_pct = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    with open(csv_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse timestamp
                t = row.get('time')
                # Supporta ISO con timezone, prova vari formati
                dt = None
                for fmt in ('%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S%z'):
                    try:
                        dt = datetime.strptime(t, fmt)
                        break
                    except Exception:
                        continue
                if dt is None:
                    # fallback: lascia stringa
                    dt = t

                eq = float(row.get('equity_est', 'nan'))
                up = float(row.get('used_pct', 'nan'))

                if eq != eq or up != up:  # NaN check
                    continue

                times.append(dt)
                equity.append(eq)
                used_pct.append(up * 100.0)  # percentuale
            except Exception:
                continue

    return times, equity, used_pct


def plot_equity(times, equity, used_pct, out_path: str, show: bool):
    if not times:
        raise ValueError("Nessun dato valido nel CSV.")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(times, equity, color='tab:blue', label='equity_est')
    ax2.plot(times, used_pct, color='tab:red', label='used_pct (%)')

    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Equity (stake currency)', color='tab:blue')
    ax2.set_ylabel('Used %', color='tab:red')

    ax1.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    # Legenda combinata
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Crea directory se non esiste
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, dpi=120)
    print(f"Grafico salvato in: {out_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot equity_est e used_pct nel tempo.')
    parser.add_argument('--csv', default=os.path.join('user_data', 'backtest_results', 'equity_progress.csv'),
                        help='Percorso del CSV generato dalla strategia.')
    parser.add_argument('--out', default=os.path.join('plot', 'equity_progress.png'),
                        help='Percorso di output del PNG del grafico.')
    parser.add_argument('--show', action='store_true', help='Mostra il grafico a schermo dopo il salvataggio.')
    args = parser.parse_args()

    times, equity, used_pct = read_equity_csv(args.csv)
    plot_equity(times, equity, used_pct, args.out, args.show)


if __name__ == '__main__':
    main()