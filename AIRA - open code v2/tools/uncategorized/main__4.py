import os
import ccxt
import time
import json
from argparse import ArgumentParser
from datetime import datetime, timedelta
import shutil
import subprocess
from pathlib import Path

# number of resulting pairs
NUM_PAIRS = 50

# percentage of pairs in next backtest cycle
# higher percentage results in long runtimes
PERC_PAIRS = 1/5

def load_backtest_data(path_to_freqtrade, exchange, timeframes, timerange):
    print(f"Load backtest data.")
    command = f"{path_to_freqtrade}.venv/bin/python"
    args = ["freqtrade", "download-data", "--exchange", exchange, "--timeframes", timeframes, "--timerange",
            f"{timerange}"]

    full_command = [command] + args

    process = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=path_to_freqtrade)

    #stdout = process.stdout
    #stderr = process.stderr
    #print("Standard Output:")
    #print(stdout)
    #print("Standard Error:")
    #print(stderr)
    print("Loading backtest data finished.")

def start_backtesting(strategy, timeframes, timerange):
    print(f"Start backtesting.")
    command = f"{path_to_freqtrade}.venv/bin/python"
    args = ["freqtrade", "backtesting", "--strategy", strategy, "--timeframe", timeframes, "--timerange",
            f"{timerange}"]

    full_command = [command] + args
    process = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=path_to_freqtrade)

    stdout = process.stdout
    stderr = process.stderr
    print("Standard Output:")
    print(stdout)
    print("Standard Error:")
    print(stderr)
    print("Backtesting finished.")


def parse_backtest_json(backtest_cycle):

    backtest_result_file = f"{path_to_freqtrade}user_data/backtest_results/.last_result.json"
    # Read the original JSON file
    with open(backtest_result_file, 'r') as file:
        backtestfile = json.load(file)

    backtest_result = f"{path_to_freqtrade}user_data/backtest_results/{backtestfile['latest_backtest']}"

    # Read the original JSON file
    with open(backtest_result, 'r') as file:
        backtest = json.load(file)

    number_of_items = len(backtest["strategy"][strategy]["results_per_pair"])
    i = 0
    market_list = []
    for results_per_pair in backtest["strategy"][strategy]["results_per_pair"]:
        i = i + 1
        market_list.append(results_per_pair['key'])
        if (NUM_PAIRS <= round(number_of_items * PERC_PAIRS) <= i) or (round(number_of_items * PERC_PAIRS) < NUM_PAIRS and i == NUM_PAIRS):
            break

    print(f"\n    {strategy} results:")
    print(f"    backtest_cycle:     {backtest_cycle}")
    print(f"    backtest_days:      {backtest['strategy'][strategy]['backtest_days']}")
    print(f"    backtest_days:      {backtest['strategy'][strategy]['backtest_days']}")
    print(f"    stake_amount:       {backtest['strategy'][strategy]['stake_amount']}")
    print(f"    number of pairs:    {backtest['strategy'][strategy]['max_open_trades_setting']}")
    print(f"    profit_total:       {backtest['strategy'][strategy]['profit_total']}")
    print(f"    total_trades:       {backtest['strategy'][strategy]['total_trades']}")
    print(f"    total_volume:       {backtest['strategy'][strategy]['total_volume']}")
    print(f"    trade_count_long:   {backtest['strategy'][strategy]['trade_count_long']}")
    print(f"    trades_per_day:     {backtest['strategy'][strategy]['trades_per_day']}")
    print(f"    wins:               {backtest['strategy'][strategy]['wins']}")
    print(f"    losses:             {backtest['strategy'][strategy]['losses']}")
    print(f"    starting_balance:   {backtest['strategy'][strategy]['starting_balance']}")
    print(f"    final_balance:      {backtest['strategy'][strategy]['final_balance']}\n\n")

    return market_list


def load_markets(exchange):

    if exchange == "binance":
        ex = ccxt.binance({'verbose': False})  # log HTTP requests
    elif exchange == "kucoin":
        ex = ccxt.kucoin({'verbose': False})  # log HTTP requests
    elif exchange == "bitrue":
        ex = ccxt.bitrue({'verbose': False})  # log HTTP requests
    else:
        raise ValueError("Exchange unknown")

    print(f"Load markets for {exchange}")
    ex.load_markets()  # request markets

    markets = list(ex.markets.keys())
    market_list = []
    for key in markets:
        if (key.__contains__("/USDT")
                and not key.__contains__("3")
                and not key.__contains__("UP")
                and not key.__contains__("DOWN")
                and not key.__contains__(":USDT")):
            market_list.append(key)

    print(f"Found {len(market_list)} pairs")
    return market_list

def create_template_config(path_to_freqtrade):

    config_path = f"{path_to_freqtrade}user_data/config.json"
    template_path = f"{path_to_freqtrade}user_data/config_template.json"
    file_path = Path(template_path)

    # check if template exists - create if not
    if file_path.exists():
        print("config_template.json for keeping all settings exists already.")
    else:
        print("config_template.json doesn't exists. To store all settings it's now created.")
        shutil.copyfile(config_path, template_path)


def read_exchange_from_config(path_to_freqtrade):

    config_path = f"{path_to_freqtrade}user_data/config.json"

    # Read the original JSON file
    with open(config_path, 'r') as file:
        config = json.load(file)
    print(f"Found exchange {config['exchange']['name']} in config file.")
    return config['exchange']['name']


def render_config(markets, path_to_freqtrade):

    config_path = f"{path_to_freqtrade}user_data/config.json"
    config_template_path = f"{path_to_freqtrade}user_data/config_template.json"

    # Read the original JSON file
    with open(config_template_path, 'r') as file:
        config = json.load(file)

    # Modify the config (Example: overwrite a field)
    config['max_open_trades'] = len(markets)
    config['stake_amount'] = 100
    config['exchange']['pair_whitelist'] = markets
    # overwrite pairlist for backtesting (other than StaticPairList is not allowed)
    config['pairlists'] = [{
        "method": "StaticPairList"
    }]

    # Write the modified config to a new file
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)


    # change content - fill market nd update max trade count

    # store/save old config with timestamp in filename

    # write new config to file
    print("Rendering config finished.")


def cleanup_config(markets, path_to_freqtrade):

    config_path = f"{path_to_freqtrade}user_data/config.json"
    config_template_path = f"{path_to_freqtrade}user_data/config_template.json"

    # Read the original JSON file
    with open(config_template_path, 'r') as file:
        config = json.load(file)

    # just update the markets template config -> live config
    # todo: order pair list alphabetically
    config['exchange']['pair_whitelist'] = sorted(markets)

    # Write the modified config to a new file
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    print("Cleanup config finished.")


def cleanup_backtest_folder(path_to_freqtrade):

    # Specify the directory path
    backtest_folder = f"{path_to_freqtrade}user_data/backtest_results"

    print(f"Clean up backtest results folder '{backtest_folder}'")

    # Iterate over all files in the directory
    for filename in os.listdir(backtest_folder):
        file_path = os.path.join(backtest_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    half_year_ago = yesterday - timedelta(days=180)

    default_range = f"{half_year_ago.strftime('%y%m%d')}-{yesterday.strftime('%y%m%d')}"

    parser = ArgumentParser()
    parser.add_argument("-tr", "--timerange", dest="timerange", default=f"{default_range}",
                        help="backtesting period", metavar="TIMERANGE")
    parser.add_argument("-tf", "--timeframes", dest="timeframes", default="5m",
                        help="timeframes of backtesting period, using 5m if not set", metavar="TIMEFRAMES")
    parser.add_argument("-strat", "--strategy", dest="strategy",
                        help="strategy to be backtested. no default.", metavar="STRATEGY")
    parser.add_argument("-p", "--path_to_freqtrade", dest="path_to_freqtrade", default="/home/tino/freqtrade/",
                        help="path to freqtrade installation", metavar="FREQTRADE")

    args = parser.parse_args()
    path_to_freqtrade = args.path_to_freqtrade
    timerange = args.timerange
    timeframes = args.timeframes
    strategy = args.strategy

    cleanup_backtest_folder(path_to_freqtrade)

    exchange = read_exchange_from_config(path_to_freqtrade)
    create_template_config(path_to_freqtrade)

    # load all available markets in that exchange
    market_list = load_markets(exchange)

    # load all timeframes for all markets
    load_backtest_data(path_to_freqtrade, exchange, timeframes, timerange)

    backtest_cycle = 0
    while len(market_list) > NUM_PAIRS:

        backtest_cycle = backtest_cycle + 1

        # update the config file with the markets
        render_config(market_list, path_to_freqtrade)

        # start backtesting
        start_backtesting(strategy, timeframes, timerange)

        # format backtest result from json
        market_list = parse_backtest_json(backtest_cycle)

        print(f"Next interation of backtesting: {len(market_list)} pairs")

    cleanup_config(market_list, path_to_freqtrade)

    print("Finished backtesting!")

