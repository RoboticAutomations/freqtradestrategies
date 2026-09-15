"""
    StarMark5mv5_1 strategy.
    此策略涉及到使用了未关闭的K线数据，因此在回测中收益惊人，切勿使用此策略进行live交易。
"""
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
from enum import Enum
from datetime import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.persistence import Trade


class StarMark5mv5_1(IStrategy):
    mode_continue_times = IntParameter(2, 4, default=2, space="buy", optimize=False)
    mode_pairs_proportion = DecimalParameter(0.2, 1.0, default=0.5, decimals=1, space="buy", optimize=False)

    ma_period = IntParameter(5, 20, default=14, space="buy", optimize=True)
    rsi_period = IntParameter(10, 30, default=14, space="buy", optimize=True)

    leverage_value = 50
    can_short: bool = True

    minimal_roi = {
        "120": 0.382,
        "60": 0.618,
        "0": 1.618,
    }

    stoploss = -0.618


    trailing_stop = True
    trailing_stop_positive = 0.037
    trailing_stop_positive_offset = 0.436
    trailing_only_offset_is_reached = True

    timeframe = "5m"

    startup_candle_count: int = 30


    class MarketMode(Enum):  
        BEAR = -1
        BULL = 1
        SIDEWAYS = 0

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '8h') for pair in pairs] + [(pair, '4h') for pair in pairs] + [(pair, '2h') for pair in pairs] + [(pair, '1h') for pair in pairs]
        return informative_pairs

    
    def give_market_mode_indicator(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe = dataframe.sort_values('date').copy()
        pair = metadata['pair']
        
        inf_timeframes = ['8h', '4h', '2h', '1h'] # 更新时间框架列表
        
        mode_tf = {}
        
        for tf in inf_timeframes:
            pair_df = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
            if pair_df is None or len(pair_df) < self.mode_continue_times.value:
                mode_tf[tf] = np.full(len(dataframe), StarMark5mv5_1.MarketMode.SIDEWAYS.value, dtype=object)
                continue

            pair_df = pair_df.sort_values('date').copy()
            pair_df['diff'] = pair_df['close'].diff()
            window = self.mode_continue_times.value - 1

            pair_df['bullish'] = (pair_df['diff'] > 0).rolling(window=window).sum() == window
            pair_df['bearish'] = (pair_df['diff'] < 0).rolling(window=window).sum() == window

            merged = pd.merge_asof(
                dataframe[['date']].copy(),
                pair_df[['date', 'bullish', 'bearish']],
                on='date',
                direction='backward'
            )
            
            valid = merged['bullish'].notna()
            valid_pairs = valid.astype(int)
            bull_signals = np.where(valid & merged['bullish'].fillna(False), 1, 0)
            bear_signals = np.where(valid & merged['bearish'].fillna(False), 1, 0)

            mode_arr = np.full(len(dataframe), StarMark5mv5_1.MarketMode.SIDEWAYS.value, dtype=object)
            valid_idx = valid_pairs > 0
            bull_prop = np.zeros(len(dataframe))
            bear_prop = np.zeros(len(dataframe))
            bull_prop[valid_idx] = bull_signals[valid_idx] / valid_pairs[valid_idx]
            bear_prop[valid_idx] = bear_signals[valid_idx] / valid_pairs[valid_idx]
            
            mode_arr[(bull_prop >= self.mode_pairs_proportion.value) & valid_idx] = StarMark5mv5_1.MarketMode.BULL.value
            mode_arr[(bear_prop >= self.mode_pairs_proportion.value) & valid_idx] = StarMark5mv5_1.MarketMode.BEAR.value
            mode_tf[tf] = mode_arr

        final_mode = np.full(len(dataframe), StarMark5mv5_1.MarketMode.SIDEWAYS.value, dtype=object)
        for i in range(len(dataframe)):
            # 检查是否所有定义的时间框架都指示BULL
            all_bull = all(mode_tf[tf][i] == StarMark5mv5_1.MarketMode.BULL.value for tf in inf_timeframes if tf in mode_tf and i < len(mode_tf[tf]))
            # 检查是否所有定义的时间框架都指示BEAR
            all_bear = all(mode_tf[tf][i] == StarMark5mv5_1.MarketMode.BEAR.value for tf in inf_timeframes if tf in mode_tf and i < len(mode_tf[tf]))

            if all_bull:
                final_mode[i] = StarMark5mv5_1.MarketMode.BULL.value
            elif all_bear:
                final_mode[i] = StarMark5mv5_1.MarketMode.BEAR.value
            else:
                final_mode[i] = StarMark5mv5_1.MarketMode.SIDEWAYS.value

        dataframe['market_mode'] = final_mode

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.give_market_mode_indicator(dataframe, metadata) # 这已经按日期排序

        pair = metadata['pair']
        inf_timeframes_for_rsi = ['8h', '4h', '2h', '1h'] # 更新RSI的时间框架列表

        for tf in inf_timeframes_for_rsi:
            inf_df = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
            if inf_df is not None and not inf_df.empty:
                inf_df = inf_df.sort_values('date').copy() # 确保为 merge_asof 排序
                inf_df[f'rsi_{tf}'] = pta.rsi(inf_df['close'], length=self.rsi_period.value)
                # 将信息性RSI合并到主数据帧中
                dataframe = pd.merge_asof(
                    dataframe, # 主数据帧，已通过 give_market_mode_indicator 按日期排序
                    inf_df[['date', f'rsi_{tf}']],
                    on='date',
                    direction='backward'
                )
                # 如果 merge_asof 在开头产生NaN或者inf_df包含NaN，则填充RSI的NaN值
                dataframe[f'rsi_{tf}'] = dataframe[f'rsi_{tf}'].bfill() # 首先向后填充
                dataframe[f'rsi_{tf}'] = dataframe[f'rsi_{tf}'].fillna(50) # 用中性值50填充剩余的NaN
                # 添加日志输出
                if not dataframe.empty:
                    print(f"Pair: {metadata['pair']}, Timeframe: {tf}, Last RSI value: {dataframe[f'rsi_{tf}'].iloc[-1]}")
            else:
                # 如果信息数据不可用，则用中性RSI值（例如50）填充
                dataframe[f'rsi_{tf}'] = 50
            #     # 添加日志输出
                if not dataframe.empty:
                    print(f"Pair: {metadata['pair']}, Timeframe: {tf}, RSI set to 50 (no data)")

        # 计算 ADX 指标
        for tf in inf_timeframes_for_rsi: # 重用RSI的时间框架列表
            inf_df = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
            if inf_df is not None and not inf_df.empty:
                if not all(col in inf_df.columns for col in ['high', 'low', 'close']):
                    dataframe[f'adx_{tf}'] = 0 # 如果缺少HLC列，则填充默认值
                    if not dataframe.empty:
                        print(f"Pair: {metadata['pair']}, Timeframe: {tf}, ADX set to 0 (missing HLC data)")
                    continue
                
                inf_df = inf_df.sort_values('date').copy() # 确保为 merge_asof 排序
                # ADX 需要高、低、收盘价。默认周期为14。
                # 确保有足够的数据进行计算，talib通常会返回NaNs如果数据不足
                adx_series = ta.ADX(inf_df['high'], inf_df['low'], inf_df['close'], timeperiod=14)
                inf_df[f'adx_{tf}'] = adx_series
                
                dataframe = pd.merge_asof(
                    dataframe,
                    inf_df[['date', f'adx_{tf}']],
                    on='date',
                    direction='backward'
                )
                dataframe[f'adx_{tf}'] = dataframe[f'adx_{tf}'].bfill() # 首先向后填充
                dataframe[f'adx_{tf}'] = dataframe[f'adx_{tf}'].fillna(0) # 用0填充剩余的NaN (表示无趋势或数据不足)

        dataframe['ma'] = pta.sma(dataframe['close'], length=self.ma_period.value)
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_period.value) # 这是5分钟RSI
        # 计算 ATR
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14) # ATR 通常使用14周期
        dataframe['volatility'] = dataframe['close'].rolling(window=20).std()
        dataframe['price_change'] = dataframe['close'].pct_change()
        
        # 计算动态止损参考
        dataframe['dynamic_stop'] = dataframe['atr'] * 2 / dataframe['close']

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 添加波动率过滤
        volatility_threshold = dataframe['volatility'].rolling(window=50).quantile(0.8)
        
        # 多头入场 - 添加波动率过滤
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['close'], dataframe['ma'])) &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['rsi'] < 70) &
            (dataframe['market_mode'] == StarMark5mv5_1.MarketMode.BULL.value) &
            (dataframe['rsi_8h'] < 75) &
            (dataframe['rsi_4h'] < 75) &
            (dataframe['adx_8h'] > 25) &
            (dataframe['adx_4h'] > 25) &
            (dataframe['adx_2h'] > 25) &
            (dataframe['volatility'] < volatility_threshold),  # 避免高波动时入场
            ['enter_long', 'enter_tag']
        ] = (1, 'long_enter')

        # 空头入场 - 添加波动率过滤
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['close'], dataframe['ma'])) &
            (dataframe['close'] < dataframe['open']) &
            (dataframe['rsi'] > 30) &
            (dataframe['market_mode'] == StarMark5mv5_1.MarketMode.BEAR.value) &
            (dataframe['rsi_8h'] > 25) &
            (dataframe['rsi_4h'] > 25) &
            (dataframe['adx_2h'] > 25) &
            (dataframe['adx_4h'] > 25) &
            (dataframe['adx_8h'] > 25) &
            (dataframe['volatility'] < volatility_threshold),  # 避免高波动时入场
            ['enter_short', 'enter_tag']
        ] = (1, 'short_enter')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 空头退出条件
        dataframe.loc[
            (dataframe['market_mode'] == StarMark5mv5_1.MarketMode.BULL.value),
            ['exit_short', 'exit_tag']
        ] = (1, 'short_exit_market')

        dataframe.loc[
            (dataframe['market_mode'] == StarMark5mv5_1.MarketMode.BEAR.value),
            ['exit_long', 'exit_tag']
        ] = (1, 'long_exit_market')


        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, **kwargs) -> float:
        """
        定义固定杠杆。
        """
        return self.leverage_value

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           **kwargs) -> float:
        """
        动态调整仓位大小，基于ATR
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return min_stake
            
        # 获取当前ATR
        current_atr = dataframe['atr'].iloc[-1]
        current_price = current_rate
        
        # 基于ATR计算风险调整后的仓位
        # 目标：每笔交易最大风险为账户的1%
        account_balance = self.wallets.get_total_stake_amount()
        max_risk_per_trade = account_balance * 0.01  # 1%风险
        
        # 计算基于ATR的止损距离
        atr_stop_distance = current_atr * 2  # 2倍ATR作为止损
        stop_loss_rate = atr_stop_distance / current_price
        
        # 考虑杠杆的影响
        effective_stop_loss = stop_loss_rate * self.leverage_value
        
        # 计算合适的仓位大小
        if effective_stop_loss > 0:
            position_size = max_risk_per_trade / effective_stop_loss
            return min(max(position_size, min_stake), max_stake)
        
        return min_stake

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        动态止损，防止liquidation
        """
        # 如果已经接近强制平仓线，提前止损
        liquidation_threshold = -0.95 / self.leverage_value  # 留5%缓冲
        
        if current_profit <= liquidation_threshold:
            return 0.01  # 立即止损
            
        # 获取数据框架
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return self.stoploss
            
        # 基于ATR动态调整止损
        current_atr = dataframe['atr'].iloc[-1]
        entry_price = trade.open_rate
        
        # 计算ATR止损
        atr_stop_distance = (current_atr * 1.5) / entry_price
        atr_stoploss = -atr_stop_distance * self.leverage_value
        
        # 使用更保守的止损
        conservative_stop = max(atr_stoploss, -0.5)  # 最大50%亏损
        
        return max(conservative_stop, liquidation_threshold)