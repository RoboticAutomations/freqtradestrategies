import dash
from dash import dcc, html, Input, Output, State
import logging
import plotly.graph_objs as go
import pandas as pd
import threading
import time
import json
import os
from datetime import datetime
from integrated_engine import IntegratedEngine

logger = logging.getLogger(__name__)

app = dash.Dash(__name__)
app.title = "DarkReaper>>Order Book Dashboard"


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .metric-card { background: white; border-radius: 10px; padding: 15px; 
                          box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                          min-width: 120px; text-align: center; flex-shrink: 0; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Span("💀", style={'fontSize': '60px', 'marginRight': '20px'}),
            ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
            html.Div([
                html.H1([
                    html.Span("DarkReaper", style={'color': '#ff0000', 'fontWeight': 'bold', 'textShadow': '2px 2px 4px rgba(0,0,0,0.5)'}),
                    html.Span(" >> ", style={'color': '#ffffff'}),
                    html.Span("Order Book Dashboard", style={'color': '#ffffff'})
                ], style={'margin': '0', 'fontSize': '36px'}),
                html.P("Real-time monitoring of the order book for multiple coins",
                       style={'margin': '5px 0 0 0', 'color': '#e0e0e0', 'fontSize': '14px'}),
            ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
    ], className='header'),

    html.Div([
        html.Button('⚙️ Settings', id='settings-toggle-btn', n_clicks=0,
                   style={'padding': '10px 20px', 'backgroundColor': '#667eea', 
                          'color': 'white', 'border': 'none', 'borderRadius': '5px', 
                          'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold',
                          'margin': '10px'}),
        html.Div(id='settings-panel', style={'display': 'none'}, children=[
            html.Div([
                html.H3("🔧 Engine Settings", style={'textAlign': 'center', 'color': '#2d3748'}),

                html.Div([
                    html.Label("📊 OrderBook Depth:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Button('25', id='depth-25-btn', n_clicks=0, 
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#48bb78', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('50', id='depth-50-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#4299e1', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('100', id='depth-100-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#ed8936', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                ], style={'padding': '10px', 'textAlign': 'center'}),

                html.Div([
                    html.Label("🔄 CVD Reset Hours:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Button('Unlimited', id='cvd-none-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#48bb78', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('24h', id='cvd-24-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#4299e1', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('48h', id='cvd-48-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#ed8936', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                ], style={'padding': '10px', 'textAlign': 'center'}),

                html.Div([
                    html.Label("🕐 Candle Interval (seconds):", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Button('60s', id='candle-60-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#48bb78', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('120s', id='candle-120-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#4299e1', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('300s', id='candle-300-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#ed8936', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                ], style={'padding': '10px', 'textAlign': 'center'}),

                html.Div([
                    html.Label("🏦 Exchange:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Button('Bybit', id='exchange-bybit-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#48bb78', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('Binance', id='exchange-binance-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#4299e1', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                ], style={'padding': '10px', 'textAlign': 'center'}),

                html.Div([
                    html.Label("⏱️ Refresh Interval (ms):", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Button('1500ms', id='refresh-1500-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#48bb78', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('3000ms', id='refresh-3000-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#4299e1', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('5000ms', id='refresh-5000-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#ed8936', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                ], style={'padding': '10px', 'textAlign': 'center'}),

                html.Div([
                    html.Label("📈 Enable Trades:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Button('ON', id='trades-on-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#48bb78', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Button('OFF', id='trades-off-btn', n_clicks=0,
                               style={'padding': '8px 15px', 'margin': '5px', 'backgroundColor': '#ed8936', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                ], style={'padding': '10px', 'textAlign': 'center'}),

                html.Div(id='settings-status', style={'textAlign': 'center', 'padding': '10px', 
                                                      'color': '#48bb78', 'fontWeight': 'bold'}),

                html.Div([
                    html.H4("Current Settings:", style={'textAlign': 'center', 'color': '#4a5568'}),
                    html.Pre(id='current-settings-display', style={
                        'backgroundColor': '#f7fafc', 'padding': '15px', 'borderRadius': '5px',
                        'fontSize': '12px', 'border': '1px solid #cbd5e0'
                    })
                ], style={'padding': '15px'}),

            ], style={'backgroundColor': '#f7fafc', 'padding': '20px', 'borderRadius': '10px', 'margin': '20px'})
        ])
    ], style={'textAlign': 'center'}),

    html.Div([

        html.Div([
            html.H3("⚙️ Symbol Management", style={'textAlign': 'center', 'color': '#2d3748', 'margin': '10px 0'}),
            html.Div([
                html.Div([
                    dcc.Input(
                        id='new-symbol-input',
                        type='text',
                        placeholder='Add symbol (e.g., ADA/USDT)',
                        style={'width': '250px', 'padding': '10px', 'marginRight': '10px', 'fontSize': '14px'}
                    ),
                    html.Button('➕ Add', id='add-symbol-btn', n_clicks=0, 
                               style={'padding': '10px 20px', 'backgroundColor': '#48bb78', 
                                      'color': 'white', 'border': 'none', 'borderRadius': '5px', 
                                      'cursor': 'pointer', 'fontSize': '14px', 'fontWeight': 'bold'}),
                ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '15px'}),
                html.Div(id='symbols-list-container', style={
                    'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 
                    'gap': '10px', 'padding': '10px'
                }),
                html.Div(id='symbol-management-status', style={'textAlign': 'center', 'padding': '10px', 'color': '#718096'}),
            ], style={'backgroundColor': '#f7fafc', 'padding': '20px', 'borderRadius': '10px', 'margin': '20px'}),
        ]),

        html.Div([
            html.Label("Select Coin:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[],  # Will be populated dynamically
                value=None,  # Initial value
                placeholder="Select a symbol",
                style={'width': '300px', 'margin': '10px'}
            ),
        ], style={'textAlign': 'center', 'padding': '20px'}),

        html.Div(id='status-indicator', style={'textAlign': 'center', 'padding': '10px'}),

        html.Div([
            html.Div(id='live-metrics', style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'center',
                'alignItems': 'stretch',
                'gap': '10px',
                'flexWrap': 'nowrap',
                'overflowX': 'auto',
                'padding': '10px'
            })
        ]),

        html.Div([
            html.H2("Advanced Market Signals",
                   style={'textAlign': 'center', 'color': '#2d3748', 'margin': '20px 0 10px 0'}),
            html.P("Order Book Analysis",
                  style={'textAlign': 'center', 'color': '#718096', 'fontSize': '14px', 'margin': '0 0 15px 0'}),
            html.Div(id='advanced-signals', style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'center',
                'alignItems': 'stretch',
                'gap': '10px',
                'flexWrap': 'nowrap',
                'overflowX': 'auto',
                'padding': '10px',
                'background': 'linear-gradient(135deg, #667eea10 0%, #764ba210 100%)',
                'borderRadius': '10px',
                'margin': '0 10px'
            })
        ]),

        html.Div([
            dcc.Graph(id='delta-chart', style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id='cvd-chart', style={'width': '48%', 'display': 'inline-block'}),
        ]),

        html.Div([
            dcc.Graph(id='liquidity-chart', style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id='pressure-chart', style={'width': '48%', 'display': 'inline-block'}),
        ]),

        html.Div([
            dcc.Graph(id='trade-delta-chart', style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(id='trade-volume-chart', style={'width': '48%', 'display': 'inline-block'}),
        ]),

        html.Div([
            html.Div([
                html.H3("🟢 Bids"),
                html.Div(id='bids-table')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H3("🔴 Asks"),
                html.Div(id='asks-table')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ]),

        dcc.Interval(
            id='interval-component',
            interval=1500,  # Initial value, will be updated by callback dynamically
            n_intervals=0
        )
    ])
])

# Global data storage
SYMBOLS_FILE = "dashboard_symbols.json"

def load_config():
    """Load symbols and settings from JSON file or return defaults"""
    default_config = {
        'symbols': ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
        'settings': {
            'depth': 25,
            'cvd_reset_hours': None,
            'max_history': 1000,
            'candle_interval': 60,
            'exchange_name': 'bybit',
            'enable_trades': True,
            'refresh_interval': 1500
        }
    }

    if os.path.exists(SYMBOLS_FILE):
        try:
            with open(SYMBOLS_FILE, 'r') as f:
                data = json.load(f)
                # Handle old format (just list of symbols)
                if isinstance(data, list):
                    return {'symbols': data, 'settings': default_config['settings']}
                # New format with settings
                if isinstance(data, dict):
                    if 'symbols' not in data:
                        data['symbols'] = default_config['symbols']
                    if 'settings' not in data:
                        data['settings'] = default_config['settings']
                    else:
                        # Merge with defaults for missing keys
                        for key, value in default_config['settings'].items():
                            if key not in data['settings']:
                                data['settings'][key] = value
                    return data
        except Exception as e:
            print(f"Error loading config: {e}")
    
    return default_config

def load_symbols():
    """Load symbols from config"""
    return load_config()['symbols']

def save_config(symbols=None, settings=None):
    """Save symbols and/or settings to JSON file"""
    try:
        # Load existing config
        config = load_config()
        
        # Update symbols if provided
        if symbols is not None:
            config['symbols'] = symbols
        
        # Update settings if provided
        if settings is not None:
            config['settings'].update(settings)
        
        # Save to file
        with open(SYMBOLS_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def save_symbols(symbols):
    """Save symbols to JSON file (backward compatibility)"""
    return save_config(symbols=symbols)

config = load_config()

global_store = {
    'engine': None,
    'symbols_history': {},
    'integrated_history': {},
    'max_history': 100,
    'available_symbols': config['symbols'],
    'settings': config['settings'],
    'update_thread': None
}

def init_engine():
    old_engine = global_store.get('engine')
    settings = global_store['settings']
    global_store['engine'] = IntegratedEngine(
        symbols=global_store['available_symbols'],
        exchange_name=settings.get('exchange_name', 'bybit'),
        candle_interval=settings.get('candle_interval', 60),
        enable_trades=settings.get('enable_trades', True),
        depth=settings.get('depth', 25),
        cvd_reset_hours=settings.get('cvd_reset_hours'),
        max_history=settings.get('max_history', 1000)
    )

    if old_engine:
        try:
            old_engine.stop()
        except:
            pass


    for symbol in global_store['available_symbols']:
        if symbol not in global_store['symbols_history']:
            global_store['symbols_history'][symbol] = []
        if symbol not in global_store['integrated_history']:
            global_store['integrated_history'][symbol] = []

    def update_loop():
        while True:
            try:
                if global_store['engine']:
                    all_metrics = global_store['engine'].update_all_orderbooks()
                    integrated_metrics = global_store['engine'].get_all_integrated_metrics()
                    for symbol in global_store['available_symbols']:
                        # Order Book metrics
                        if symbol in all_metrics and all_metrics[symbol]:
                            metrics = all_metrics[symbol]

                            if symbol not in global_store['symbols_history']:
                                global_store['symbols_history'][symbol] = []

                            global_store['symbols_history'][symbol].append(metrics)
                            if len(global_store['symbols_history'][symbol]) > global_store['max_history']:
                                global_store['symbols_history'][symbol].pop(0)

                        if symbol in integrated_metrics and integrated_metrics[symbol]:
                            int_metrics = integrated_metrics[symbol]

                            if symbol not in global_store['integrated_history']:
                                global_store['integrated_history'][symbol] = []

                            global_store['integrated_history'][symbol].append(int_metrics)

                            # Limit history
                            if len(global_store['integrated_history'][symbol]) > global_store['max_history']:
                                global_store['integrated_history'][symbol].pop(0)
            except Exception as e:
                logging.error(f"Error in update loop: {e}")

            time.sleep(1)

    if not global_store.get('update_thread') or not global_store['update_thread'].is_alive():
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        global_store['update_thread'] = thread
        logger.info("Continuous capture thread started ")


init_engine()

def interpret_imbalance_velocity(velocity: float) -> tuple[str, str]:
    if velocity > 0.5:
        return "BULLISH", "#48bb78"
    elif velocity < -0.5:
        return "BEARISH", "#f56565"
    else:
        return "NEUTRAL", "#a0aec0"

def interpret_entropy(entropy: float) -> tuple[str, str]:
    if entropy < 0.3:
        return "STABLE", "#48bb78"
    elif entropy > 0.7:
        return "CHAOTIC", "#f56565"
    else:
        return "NORMAL", "#a0aec0"

def interpret_weighted_mid_price(wmp: float, mid_price: float) -> tuple[str, str]:
    if mid_price <= 0:
        return "N/A", "#a0aec0"

    diff_pct = abs(wmp - mid_price) / mid_price

    if diff_pct > 0.005:  # 0.5%
        if wmp > mid_price:
            return "BULLISH", "#48bb78"
        else:
            return "BEARISH", "#f56565"
    else:
        return "BALANCED", "#a0aec0"

def interpret_liquidity_imbalance(ratio: float) -> tuple[str, str]:
    if ratio > 1.5:
        return "BULLISH", "#48bb78"
    elif ratio < 0.67:
        return "BEARISH", "#f56565"
    else:
        return "BALANCED", "#a0aec0"

def interpret_smart_money_index(smi: float) -> tuple[str, str]:
    if smi > 0.7:
        return "WHALES", "#9f7aea"
    elif smi > 0.4:
        return "MIXED", "#ed8936"
    else:
        return "RETAIL", "#a0aec0"

def interpret_price_level_momentum(momentum: float) -> tuple[str, str]:
    if momentum > 0.5:
        return "BULLISH", "#48bb78"
    elif momentum < -0.5:
        return "BEARISH", "#f56565"
    else:
        return "NEUTRAL", "#a0aec0"

def interpret_support_resistance_strength(strength: float, level_type: str) -> tuple[str, str]:
    if strength > 3.0:
        return f"STRONG", "#9f7aea"
    elif strength > 1.5:
        return f"MODERATE", "#ed8936"
    else:
        return f"WEAK", "#a0aec0"

def interpret_depth_asymmetry(asymmetry: float) -> tuple[str, str]:
    if asymmetry > 0.5:
        return "BID HEAVY", "#48bb78"
    elif asymmetry < -0.5:
        return "ASK HEAVY", "#f56565"
    else:
        return "SYMMETRIC", "#a0aec0"

def interpret_liquidity_fragmentation(fragmentation: float) -> tuple[str, str]:
    if fragmentation > 0.75:
        return "FRAGMENTED", "#f56565"
    elif fragmentation < 0.25:
        return "CONCENTRATED", "#9f7aea"
    else:
        return "MODERATE", "#a0aec0"

def interpret_spread_volatility(spread_vol: float, history) -> tuple[str, str]:
    if len(history) < 10:
        return "N/A", "#a0aec0"

    recent_spreads = [m.spread_volatility for m in history[-20:]]
    mean_vol = sum(recent_spreads) / len(recent_spreads) if recent_spreads else 0
    if spread_vol > mean_vol * 2:
        return "HIGH VOL", "#f56565"
    elif spread_vol < mean_vol * 0.5:
        return "LOW VOL", "#48bb78"
    else:
        return "NORMAL", "#a0aec0"

def create_advanced_signal_card(title: str, signal: str, color: str, description: str = "") -> html.Div:
    return html.Div([
        html.H5(title, style={
            'margin': '0 0 5px 0',
            'fontSize': '12px',
            'color': '#4a5568',
            'fontWeight': '600',
            'textTransform': 'uppercase'
        }),
        html.H3(signal, style={
            'margin': '0', 
            'fontSize': '16px',
            'color': color,
            'fontWeight': 'bold'
        }),
        html.P(description, style={
            'margin': '5px 0 0 0',
            'fontSize': '10px',
            'color': '#718096'
        }) if description else None
    ], className='metric-card', style={
        'minWidth': '140px',
        'textAlign': 'center',
        'padding': '12px',
        'border': f'2px solid {color}',
        'borderRadius': '8px'
    })


init_engine()

@app.callback(
    Output('settings-panel', 'style'),
    Input('settings-toggle-btn', 'n_clicks')
)
def toggle_settings(n_clicks):
    """Toggle settings panel visibility"""
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output('current-settings-display', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_settings_display(n):
    """Display current settings"""
    settings = global_store['settings']
    cvd_display = "Unlimited" if settings.get('cvd_reset_hours') is None else f"{settings.get('cvd_reset_hours')}h"
    return f"""Depth: {settings.get('depth', 25)} levels
CVD Reset: {cvd_display}
Candle Interval: {settings.get('candle_interval', 60)}s
Exchange: {settings.get('exchange_name', 'bybit').upper()}
Refresh Interval: {settings.get('refresh_interval', 1500)}ms
Enable Trades: {'ON' if settings.get('enable_trades', True) else 'OFF'}
Max History: {settings.get('max_history', 1000)} candles"""

@app.callback(
    Output('settings-status', 'children'),
    [Input('depth-25-btn', 'n_clicks'),
     Input('depth-50-btn', 'n_clicks'),
     Input('depth-100-btn', 'n_clicks'),
     Input('cvd-none-btn', 'n_clicks'),
     Input('cvd-24-btn', 'n_clicks'),
     Input('cvd-48-btn', 'n_clicks'),
     Input('candle-60-btn', 'n_clicks'),
     Input('candle-120-btn', 'n_clicks'),
     Input('candle-300-btn', 'n_clicks'),
     Input('exchange-bybit-btn', 'n_clicks'),
     Input('exchange-binance-btn', 'n_clicks'),
     Input('refresh-1500-btn', 'n_clicks'),
     Input('refresh-3000-btn', 'n_clicks'),
     Input('refresh-5000-btn', 'n_clicks'),
     Input('trades-on-btn', 'n_clicks'),
     Input('trades-off-btn', 'n_clicks')],
    prevent_initial_call=True
)
def handle_settings_buttons(*args):
    """Handle all settings button clicks"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    settings_changed = False
    message = ""
    
    # Depth settings
    if button_id == 'depth-25-btn':
        global_store['settings']['depth'] = 25
        settings_changed = True
        message = "✅ OrderBook Depth set to 25 levels"
    elif button_id == 'depth-50-btn':
        global_store['settings']['depth'] = 50
        settings_changed = True
        message = "✅ OrderBook Depth set to 50 levels"
    elif button_id == 'depth-100-btn':
        global_store['settings']['depth'] = 100
        settings_changed = True
        message = "✅ OrderBook Depth set to 100 levels"
    
    # CVD Reset settings
    elif button_id == 'cvd-none-btn':
        global_store['settings']['cvd_reset_hours'] = None
        settings_changed = True
        message = "✅ CVD Reset: UNLIMITED (never resets)"
    elif button_id == 'cvd-24-btn':
        global_store['settings']['cvd_reset_hours'] = 24
        settings_changed = True
        message = "✅ CVD Reset: Every 24 hours"
    elif button_id == 'cvd-48-btn':
        global_store['settings']['cvd_reset_hours'] = 48
        settings_changed = True
        message = "✅ CVD Reset: Every 48 hours"
    
    # Candle Interval settings
    elif button_id == 'candle-60-btn':
        global_store['settings']['candle_interval'] = 60
        settings_changed = True
        message = "✅ Candle Interval: 60 seconds"
    elif button_id == 'candle-120-btn':
        global_store['settings']['candle_interval'] = 120
        settings_changed = True
        message = "✅ Candle Interval: 120 seconds"
    elif button_id == 'candle-300-btn':
        global_store['settings']['candle_interval'] = 300
        settings_changed = True
        message = "✅ Candle Interval: 300 seconds (5 min)"
    
    # Exchange settings
    elif button_id == 'exchange-bybit-btn':
        global_store['settings']['exchange_name'] = 'bybit'
        settings_changed = True
        message = "✅ Exchange: Bybit"
    elif button_id == 'exchange-binance-btn':
        global_store['settings']['exchange_name'] = 'binance'
        settings_changed = True
        message = "✅ Exchange: Binance"
    
    # Refresh Interval settings
    elif button_id == 'refresh-1500-btn':
        global_store['settings']['refresh_interval'] = 1500
        settings_changed = True
        message = "✅ Refresh Interval: 1500ms (1.5s)"
    elif button_id == 'refresh-3000-btn':
        global_store['settings']['refresh_interval'] = 3000
        settings_changed = True
        message = "✅ Refresh Interval: 3000ms (3s)"
    elif button_id == 'refresh-5000-btn':
        global_store['settings']['refresh_interval'] = 5000
        settings_changed = True
        message = "✅ Refresh Interval: 5000ms (5s)"
    
    # Enable Trades settings
    elif button_id == 'trades-on-btn':
        global_store['settings']['enable_trades'] = True
        settings_changed = True
        message = "✅ Trade Collection: ENABLED"
    elif button_id == 'trades-off-btn':
        global_store['settings']['enable_trades'] = False
        settings_changed = True
        message = "✅ Trade Collection: DISABLED"
    
    # Save settings if changed
    if settings_changed:
        save_config(settings=global_store['settings'])
        # Reinitialize engine with new settings
        init_engine()
        message += " | Engine reinitialized with new settings"
    
    return message

@app.callback(
    Output('interval-component', 'interval'),
    Input('settings-status', 'children')
)
def update_refresh_interval(status_message):
    """Update refresh interval dynamically when settings change"""
    return global_store['settings'].get('refresh_interval', 1500)

@app.callback(
    Output('symbols-list-container', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_symbols_list(n):
    symbols_cards = []
    for symbol in global_store['available_symbols']:
        card = html.Div([
            html.Span(symbol, style={'marginRight': '8px', 'fontWeight': 'bold', 'color': '#2d3748'}),
            html.Button('❌', id={'type': 'remove-symbol', 'index': symbol}, n_clicks=0,
                       style={'backgroundColor': '#f56565', 'color': 'white', 'border': 'none',
                              'borderRadius': '3px', 'padding': '2px 8px', 'cursor': 'pointer',
                              'fontSize': '12px'})
        ], style={'backgroundColor': 'white', 'padding': '8px 12px', 'borderRadius': '5px',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'flex', 'alignItems': 'center'})
        symbols_cards.append(card)
    return symbols_cards

@app.callback(
    Output('symbol-dropdown', 'options'),
    Output('symbol-dropdown', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_dropdown(n):
    if n == 0:
        options = [{'label': symbol, 'value': symbol} for symbol in global_store['available_symbols']]
        return options, global_store['available_symbols'][0] if global_store['available_symbols'] else None

    options = [{'label': symbol, 'value': symbol} for symbol in global_store['available_symbols']]
    return options, dash.no_update

@app.callback(
    Output('symbol-management-status', 'children'),
    Output('new-symbol-input', 'value'),
    Input('add-symbol-btn', 'n_clicks'),
    State('new-symbol-input', 'value'),
    prevent_initial_call=True
)
def add_symbol(n_clicks, new_symbol):
    if not new_symbol or not new_symbol.strip():
        return "⚠️ Please enter a valid symbol", ""

    new_symbol = new_symbol.strip().upper()

    if new_symbol in global_store['available_symbols']:
        return f" {new_symbol} already exists", ""


    global_store['available_symbols'].append(new_symbol)

    if save_symbols(global_store['available_symbols']):

        init_engine()
        return f"✅ Added {new_symbol} successfully! Reloading engine...", ""
    else:
        global_store['available_symbols'].remove(new_symbol)
        return f"Failed to save {new_symbol}", ""

@app.callback(
    Output('symbol-management-status', 'children', allow_duplicate=True),
    Input({'type': 'remove-symbol', 'index': dash.dependencies.ALL}, 'n_clicks'),
    State({'type': 'remove-symbol', 'index': dash.dependencies.ALL}, 'id'),
    prevent_initial_call=True
)
def remove_symbol(n_clicks_list, ids_list):
    """Remove a symbol from the list"""
    if not any(n_clicks_list):
        return dash.no_update

    clicked_idx = next((i for i, clicks in enumerate(n_clicks_list) if clicks), None)
    if clicked_idx is None:
        return dash.no_update

    symbol_to_remove = ids_list[clicked_idx]['index']

    if len(global_store['available_symbols']) <= 1:
        return " Cannot remove last symbol!"

    if symbol_to_remove in global_store['available_symbols']:
        global_store['available_symbols'].remove(symbol_to_remove)

        if save_symbols(global_store['available_symbols']):

            init_engine()
            return f"Removed {symbol_to_remove} successfully! Reloading engine..."
        else:
            global_store['available_symbols'].append(symbol_to_remove)
            return f"Failed to remove {symbol_to_remove}"

    return dash.no_update

@app.callback(
    [Output('status-indicator', 'children'),
     Output('live-metrics', 'children'),
     Output('advanced-signals', 'children'),
     Output('delta-chart', 'figure'),
     Output('cvd-chart', 'figure'),
     Output('liquidity-chart', 'figure'),
     Output('pressure-chart', 'figure'),
     Output('trade-delta-chart', 'figure'),
     Output('trade-volume-chart', 'figure'),
     Output('bids-table', 'children'),
     Output('asks-table', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value')]
)
def update_dashboard(n, selected_symbol):

    if not selected_symbol or selected_symbol not in global_store['symbols_history']:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Select simbols")
        return (
            html.Div("Select Simbols", style={'color': 'orange'}),
            [],
            [],
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            "No data", "No data"
        )

    history = global_store['symbols_history'][selected_symbol]
    integrated_history = global_store['integrated_history'].get(selected_symbol, [])

    if not history:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Așteptare date pentru {selected_symbol}...")
        return (
            html.Div(f" Loading {selected_symbol}...", style={'color': 'blue'}),
            [],
            [],
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            "Loading...", "Loading..."
        )

    latest = history[-1]
    latest_integrated = integrated_history[-1] if integrated_history else None

    status_parts = [f"✅ Live - {selected_symbol} ({len(history)} OB points"]
    if latest_integrated and latest_integrated.trade_delta is not None:
        status_parts.append(f" | Trade Delta: {latest_integrated.trade_delta:.2f}")
    status_parts.append(")")
    status_text = "".join(status_parts)
    status_indicator = html.Div(status_text, style={'color': 'green', 'fontWeight': 'bold'})

    metrics_cards = [
        html.Div([
            html.H4("Symbol", style={'margin': '0'}),
            html.H2(f"{selected_symbol}", style={'color': '#667eea', 'margin': '0', 'fontSize': '18px'})
        ], className='metric-card'),

        html.Div([
            html.H4("OB Delta", style={'margin': '0'}),
            html.H2(f"{latest.delta:.2f}", style={'color': '#667eea', 'margin': '0'})
        ], className='metric-card'),
    ]

    if latest_integrated and latest_integrated.trade_delta is not None:
        metrics_cards.append(
            html.Div([
                html.H4("Trade Δ", style={'margin': '0'}),
                html.H2(f"{latest_integrated.trade_delta:.2f}", 
                       style={'color': '#38b2ac', 'margin': '0'})
            ], className='metric-card')
        )

    metrics_cards.extend([
        html.Div([
            html.H4("CVD", style={'margin': '0'}),
            html.H2(f"{latest.cvd:.2f}", style={'color': '#764ba2', 'margin': '0'})
        ], className='metric-card'),

        html.Div([
            html.H4("Spread", style={'margin': '0'}),
            html.H2(f"{latest.spread:.2f}", style={'color': '#f56565', 'margin': '0'})
        ], className='metric-card'),

        html.Div([
            html.H4("OBI", style={'margin': '0'}),
            html.H2(f"{latest.obi:.4f}", style={'color': '#48bb78', 'margin': '0'})
        ], className='metric-card'),

        html.Div([
            html.H4("Pressure", style={'margin': '0'}),
            html.H2(f"{latest.order_book_pressure:.4f}",
                   style={'color': '#ed8936', 'margin': '0'})
        ], className='metric-card'),

        html.Div([
            html.H4("Liquidity", style={'margin': '0'}),
            html.H2(f"{latest.liquidity_score:.0f}", 
                   style={'color': '#4299e1', 'margin': '0'})
        ], className='metric-card'),

        html.Div([
            html.H4("Vol. Conc.", style={'margin': '0'}),
            html.H2(f"{latest.volume_concentration:.2%}", 
                   style={'color': '#9f7aea', 'margin': '0'})
        ], className='metric-card'),
    ])

    metrics_display = metrics_cards

    advanced_signals_cards = []

    velocity_signal, velocity_color = interpret_imbalance_velocity(latest.imbalance_velocity)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Velocity",
            velocity_signal,
            velocity_color,
            f"{latest.imbalance_velocity:.3f}"
        )
    )

    entropy_signal, entropy_color = interpret_entropy(latest.order_book_entropy)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Market State",
            entropy_signal,
            entropy_color,
            f"Entropy: {latest.order_book_entropy:.2f}"
        )
    )

    wmp_signal, wmp_color = interpret_weighted_mid_price(
        latest.weighted_mid_price,
        latest.mid_price
    )
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "WMP Signal",
            wmp_signal,
            wmp_color,
            f"Δ{((latest.weighted_mid_price - latest.mid_price) / latest.mid_price * 100):.2f}%"
        )
    )

    liq5_signal, liq5_color = interpret_liquidity_imbalance(latest.liquidity_imbalance_5)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Liq Depth 5",
            liq5_signal,
            liq5_color,
            f"Ratio: {latest.liquidity_imbalance_5:.2f}"
        )
    )
    liq10_signal, liq10_color = interpret_liquidity_imbalance(latest.liquidity_imbalance_10)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Liq Depth 10",
            liq10_signal,
            liq10_color,
            f"Ratio: {latest.liquidity_imbalance_10:.2f}"
        )
    )

    smi_signal, smi_color = interpret_smart_money_index(latest.smart_money_index)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Smart Money",
            smi_signal,
            smi_color,
            f"Index: {latest.smart_money_index:.2f}"
        )
    )

    momentum_signal, momentum_color = interpret_price_level_momentum(latest.price_level_momentum)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Momentum",
            momentum_signal,
            momentum_color,
            f"{latest.price_level_momentum:.3f}"
        )
    )

    support_signal, support_color = interpret_support_resistance_strength(
        latest.support_strength, "Support"
    )
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Support",
            support_signal,
            support_color,
            f"Strength: {latest.support_strength:.2f}"
        )
    )

    resistance_signal, resistance_color = interpret_support_resistance_strength(
        latest.resistance_strength, "Resistance"
    )
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Resistance",
            resistance_signal,
            resistance_color,
            f"Strength: {latest.resistance_strength:.2f}"
        )
    )

    asymmetry_signal, asymmetry_color = interpret_depth_asymmetry(latest.depth_asymmetry)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Depth Balance",
            asymmetry_signal,
            asymmetry_color,
            f"{latest.depth_asymmetry:.3f}"
        )
    )

    frag_signal, frag_color = interpret_liquidity_fragmentation(latest.liquidity_fragmentation)
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Liquidity",
            frag_signal,
            frag_color,
            f"Frag: {latest.liquidity_fragmentation:.2f}"
        )
    )

    spread_vol_signal, spread_vol_color = interpret_spread_volatility(
        latest.spread_volatility, history
    )
    advanced_signals_cards.append(
        create_advanced_signal_card(
            "Spread Vol",
            spread_vol_signal,
            spread_vol_color,
            f"{latest.spread_volatility:.2f}"
        )
    )

    times = [datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S') for m in history]
    deltas = [m.delta for m in history]
    cvds = [m.cvd for m in history]
    liquidities = [m.liquidity_score for m in history]
    pressures = [m.order_book_pressure for m in history]

    trade_deltas = []
    trade_volumes = []
    buy_volumes = []
    sell_volumes = []

    if integrated_history:
        trade_deltas = [m.trade_delta if m.trade_delta is not None else 0 
                       for m in integrated_history]
        trade_volumes = [m.trade_volume if m.trade_volume is not None else 0 
                        for m in integrated_history]
        buy_volumes = [m.buy_volume if m.buy_volume is not None else 0 
                      for m in integrated_history]
        sell_volumes = [m.sell_volume if m.sell_volume is not None else 0 
                       for m in integrated_history]

    fig_delta = go.Figure(
        data=[go.Scatter(
            x=times, y=deltas,
            mode='lines+markers',
            name='Delta',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        )]
    )
    fig_delta.update_layout(
        title=f'Delta Volume - {selected_symbol}',
        xaxis_title='Time',
        yaxis_title='Delta',
        hovermode='x unified',
        template='plotly_white'
    )

    fig_cvd = go.Figure(
        data=[go.Scatter(
            x=times, y=cvds, 
            mode='lines', 
            name='CVD',
            line=dict(color='#764ba2', width=3),
            fill='tozeroy'
        )]
    )
    fig_cvd.update_layout(
        title=f'Cumulative Volume Delta - {selected_symbol}',
        xaxis_title='Time',
        yaxis_title='CVD',
        hovermode='x unified',
        template='plotly_white'
    )

    fig_liquidity = go.Figure(
        data=[go.Scatter(
            x=times, y=liquidities,
            mode='lines',
            name='Liquidity',
            line=dict(color='#4299e1', width=2)
        )]
    )
    fig_liquidity.update_layout(
        title=f'Liquidity Score - {selected_symbol}',
        xaxis_title='Time',
        yaxis_title='Liquidity',
        hovermode='x unified',
        template='plotly_white'
    )

    fig_pressure = go.Figure(
        data=[go.Scatter(
            x=times, y=pressures,
            mode='lines+markers',
            name='Pressure',
            line=dict(color='#ed8936', width=2),
            marker=dict(size=4)
        )]
    )
    fig_pressure.update_layout(
        title=f'Order Book Pressure - {selected_symbol}',
        xaxis_title='Time',
        yaxis_title='Pressure (-1 to 1)',
        hovermode='x unified',
        template='plotly_white'
    )

    fig_trade_delta = go.Figure()
    if trade_deltas and any(trade_deltas):
        fig_trade_delta.add_trace(go.Bar(
            x=times[-len(trade_deltas):],
            y=trade_deltas,
            name='Trade Delta',
            marker=dict(
                color=trade_deltas,
                colorscale='RdYlGn',
                cmin=-max(abs(min(trade_deltas)), abs(max(trade_deltas))),
                cmax=max(abs(min(trade_deltas)), abs(max(trade_deltas)))
            )
        ))
    fig_trade_delta.update_layout(
        title=f'Trade Delta (Buy - Sell) - {selected_symbol}',
        xaxis_title='Time',
        yaxis_title='Trade Delta',
        hovermode='x unified',
        template='plotly_white'
    )

    fig_trade_volume = go.Figure()
    if trade_volumes and any(trade_volumes):
        fig_trade_volume.add_trace(go.Scatter(
            x=times[-len(buy_volumes):],
            y=buy_volumes,
            name='Buy Volume',
            fill='tozeroy',
            line=dict(color='#48bb78', width=2)
        ))
        fig_trade_volume.add_trace(go.Scatter(
            x=times[-len(sell_volumes):],
            y=sell_volumes,
            name='Sell Volume',
            fill='tozeroy',
            line=dict(color='#f56565', width=2)
        ))
    fig_trade_volume.update_layout(
        title=f'Buy vs Sell Volume - {selected_symbol}',
        xaxis_title='Time',
        yaxis_title='Volume',
        hovermode='x unified',
        template='plotly_white'
    )

    # 3.!!!!!!!!!!!!!! Tabele order book LEVELS!!!!!
    bids_data = latest.bids[:20]
    asks_data = latest.asks[:20]

    bids_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Price", style={'textAlign': 'right', 'padding': '8px'}), 
            html.Th("Volume", style={'textAlign': 'right', 'padding': '8px'})
        ])),
        html.Tbody([
            html.Tr([
                html.Td(f"{row[0]:.2f}", style={'textAlign': 'right', 'padding': '8px', 'color': '#48bb78'}), 
                html.Td(f"{row[1]:.4f}", style={'textAlign': 'right', 'padding': '8px'})
            ])
            for row in bids_data
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})

    asks_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Price", style={'textAlign': 'right', 'padding': '8px'}), 
            html.Th("Volume", style={'textAlign': 'right', 'padding': '8px'})
        ])),
        html.Tbody([
            html.Tr([
                html.Td(f"{row[0]:.2f}", style={'textAlign': 'right', 'padding': '8px', 'color': '#f56565'}), 
                html.Td(f"{row[1]:.4f}", style={'textAlign': 'right', 'padding': '8px'})
            ])
            for row in asks_data
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})

    return (
        status_indicator,
        metrics_display,
        advanced_signals_cards,
        fig_delta, fig_cvd, fig_liquidity, fig_pressure,
        fig_trade_delta, fig_trade_volume,
        bids_table, asks_table
    )


if __name__ == '__main__':
    print("="*60)
    print("Starting INTEGRATED Multi-Symbol Dashboard")
    print("="*60)
    print(f"Available symbols: {', '.join(global_store['available_symbols'])}")
    print(f"Symbols loaded from: {SYMBOLS_FILE}")
    print("\n UNLIMITED CAPTURE MODE - No automatic resets!")
    print("="*60)
    print("Features:")
    print("  Symbol Management (Add/Remove coins)")
    print("  Auto-save symbols (persistent across restarts)")
    print("  UNLIMITED OrderBook Capture (CVD never resets)")
    print("  Continuous Thread (runs forever)")
    print("  Order Book Analysis (depth25)")
    print("  Trade Flow Analysis (aggTrade)")
    print("  SQLite Storage (orderbook_cache.db)")
    print("  Trade Delta Calculation (Buy - Sell)")
    print("  12 Advanced Signals (Auto-Interpreted)")
    print("\nAdvanced Signals:")
    print("  Velocity, Entropy, Weighted Mid Price")
    print("  Liquidity Imbalance, Smart Money Index")
    print("  Momentum, Support/Resistance Strength")
    print("  Depth Asymmetry, Fragmentation, Spread Vol")
    print("\nDashboard URL: http://localhost:8050")
    print("="*60)
    print("CVD accumulates FOREVER (no 24h reset)")
    print("Capture thread runs CONTINUOUSLY (unlimited)")
    print("="*60)
    app.run(debug=True, port=8050)