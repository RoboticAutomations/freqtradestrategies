import uuid
import json

DOCKER_COMPOSE = '''
---
services:
'''

DOCKER_STRATEGY = '''
  {strategy_name}:
    image: {strategy_image}
    restart: unless-stopped
    container_name: {strategy_name}
    volumes:
      - './user_data:/freqtrade/user_data'
    command: >
      trade
      --logfile /freqtrade/user_data/logs/freqtrade.log
      --db-url sqlite:////freqtrade/user_data/{strategy_name}.sqlite
      --config /freqtrade/user_data/{strategy_name}.json
      --strategy {strategy_name}\n
'''

CONFIG = {
    'stoploss_on_exchange': True,
    'cancel_open_orders_on_exit': False,
    'unfilledtimeout': {
        'unit': 'minutes',
        'enter': 10,
        'exit': 30
    },
    'order_types': {
        'entry': 'limit',
        'exit': 'limit',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60
    },
    'entry_pricing': {
        'price_side': 'same',
        'ask_last_balance': 0,
        'use_order_book': True,
        'order_book_top': 1,
        'price_last_balance': 0.0,
        'check_depth_of_market': {
            'enabled': False,
            'bids_to_ask_delta': 1
        }
    },
    'exit_pricing': {
        'price_side': 'same',
        'use_order_book': True,
        'order_book_top': 1
    },
    'exchange': {
        'name': '',
        'key': '',
        'secret': '',
        'ccxt_config': {
            'enableRateLimit': True
        },
        'ccxt_async_config': {
            'enableRateLimit': True,
            'rateLimit': 200
        },
        'pair_whitelist': [],
        'pair_blacklist': []
    },
    'pairlists': [
        {
            'method': 'StaticPairList'
        },
        {
            'method': 'PriceFilter',
            'min_price': 0.05,
            'low_price_ratio': 0.01
        }
    ],
    'edge': {
        'enabled': False,
        'process_throttle_secs': 3600,
        'calculate_since_number_of_days': 7,
        'allowed_risk': 0.01,
        'stoploss_range_min': -0.01,
        'stoploss_range_max': -0.1,
        'stoploss_range_step': -0.01,
        'minimum_winrate': 0.6,
        'minimum_expectancy': 0.2,
        'min_trade_number': 10,
        'max_trade_duration_minute': 1440,
        'remove_pumps': False
    },
    'telegram': {
        'enabled': True,
        'token': '',
        'chat_id': '',
        'keyboard': [ 
            [ '/daily', '/weekly', '/monthly' ], 
            [ '/start', '/stop', '/status table' ], 
            [ '/entries', '/exits' ], 
            [ '/whitelist', '/blacklist' ], 
            [ '/show_config', '/reload_config', '/help' ]
        ]
    },
    'bot_name': 'freqtrade',
    'initial_state': 'running',
    'force_entry_enable': False,
    'internals': {
        'process_throttle_secs': 5
    },
    'freqai': {
        'enabled': True,
        'purge_old_models': 2,
        'train_period_days': 15,
        'backtest_period_days': 7,
        'live_retrain_hours': 0,
        'identifier': 'unique-id',
        'feature_parameters': {
            'include_timeframes': [ '5m', '15m', '1h' ],
            'include_corr_pairlist': [ 'BTC/USDT:USDT', 'ETH/USDT:USDT' ],
            'label_period_candles': 20,
            'include_shifted_candles': 2,
            'DI_threshold': 0.9,
            'weight_factor': 0.9,
            'principal_component_analysis': False,
            'use_SVM_to_remove_outliers': True,
            'indicator_periods_candles': [ 10, 20 ],
            'plot_feature_importances': 0
        },
        'data_split_parameters': { 'test_size': 0.33, 'random_state': 1 },
        'model_training_parameters': {}
    }
}

STRATEGIES = {
    'RsiquiV4' : {
        'dry_run': True,
        'dry_run_wallet': 100,
        'stake_amount': 10,
        'max_open_trades': -1,
        'timeframe': '5m',
        'tradable_balance_ratio': 0.9,
        'trading_mode': 'futures',
        'margin_mode': 'isolated',
        'stake_currency': 'USDT',
        'fiat_display_currency': 'USD',
        'exchange_name' : 'binance',
        'exchange_key' : '',
        'exchange_secret' : '',
        'exchange_pair_blacklist' : [],
        'exchange_pair_whitelist' : [ 'BTC/USDT:USDT', 'ETH/USDT:USDT', 'FTM/USDT:USDT', 'SOL/USDT:USDT', 'ZIL/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', 'SAND/USDT:USDT',],
        'telegram' : True,
        'telegram_token' : '',
        'telegram_chat_id' : '',
        'freqai' : False,
    },
    'NOTankAi_15' : { 
        'dry_run': True,
        'dry_run_wallet': 100,
        'stake_amount': 10,
        'max_open_trades': -1,
        'timeframe': '5m',
        'tradable_balance_ratio': 0.9,
        'trading_mode': 'futures',
        'margin_mode': 'isolated',
        'stake_currency': 'USDT',
        'fiat_display_currency': 'USD',
        'exchange_name' : 'binance',
        'exchange_key' : '',
        'exchange_secret' : '',
        'exchange_pair_blacklist' : [],
        'exchange_pair_whitelist' : [ 'BTC/USDT:USDT', 'ETH/USDT:USDT', 'FTM/USDT:USDT', 'SOL/USDT:USDT', 'ZIL/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', 'SAND/USDT:USDT',],
        'telegram' : True,
        'telegram_token' : '',
        'telegram_chat_id' : '',
        'freqai' : True,
    },
    'Insomnia_short' : { 
        'dry_run': True,
        'dry_run_wallet': 100,
        'stake_amount': 10,
        'max_open_trades': -1,
        'timeframe': '5m',
        'tradable_balance_ratio': 0.9,
        'trading_mode': 'futures',
        'margin_mode': 'isolated',
        'stake_currency': 'USDT',
        'fiat_display_currency': 'USD',
        'exchange_name' : 'binance',
        'exchange_key' : '',
        'exchange_secret' : '',
        'exchange_pair_blacklist' : [],
        'exchange_pair_whitelist' : [ 'BTC/USDT:USDT', 'ETH/USDT:USDT', 'FTM/USDT:USDT', 'SOL/USDT:USDT', 'ZIL/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', 'SAND/USDT:USDT',],
        'telegram' : True,
        'telegram_token' : '',
        'telegram_chat_id' : '',
        'freqai' : False,
    },
}

for strategy_name, strategy_values in STRATEGIES.items():
    custom_config = CONFIG
    custom_config['dry_run'] = strategy_values['dry_run']
    custom_config['dry_run_wallet'] = strategy_values['dry_run_wallet']
    custom_config['stake_amount'] = strategy_values['stake_amount']
    custom_config['max_open_trades'] = strategy_values['max_open_trades']
    custom_config['timeframe'] = strategy_values['timeframe']
    custom_config['tradable_balance_ratio'] = strategy_values['tradable_balance_ratio']
    custom_config['trading_mode'] = strategy_values['trading_mode']
    custom_config['margin_mode'] = strategy_values['margin_mode']
    custom_config['stake_currency'] = strategy_values['stake_currency']
    custom_config['fiat_display_currency'] = strategy_values['fiat_display_currency']
    custom_config['exchange']['name'] = strategy_values['exchange_name']
    custom_config['exchange']['key'] = strategy_values['exchange_key']
    custom_config['exchange']['secret'] = strategy_values['exchange_secret']
    custom_config['exchange']['pair_whitelist'] = strategy_values['exchange_pair_blacklist']
    custom_config['exchange']['pair_whitelist'] = strategy_values['exchange_pair_whitelist']
    custom_config['telegram']['enabled'] = strategy_values['telegram']
    custom_config['telegram']['token'] = strategy_values['telegram_token']
    custom_config['telegram']['chat_id'] = strategy_values['telegram_chat_id']
    custom_config['freqai']['enabled'] = strategy_values['freqai']
    custom_config['freqai']['identifier'] = str(uuid.uuid4())

    with open(f'user_data/{strategy_name}.json', 'w') as file:
        json.dump(custom_config, file, indent=4)

    strategy_image = 'freqtradeorg/freqtrade:stable' if not strategy_values['freqai'] else 'freqtradeorg/freqtrade:stable_freqaitorch'

    custom_docker = DOCKER_STRATEGY.format(strategy_name=strategy_name, strategy_image=strategy_image) 
    DOCKER_COMPOSE = DOCKER_COMPOSE + custom_docker

with open('docker-compose.yml', 'w') as f:
    f.write(DOCKER_COMPOSE)