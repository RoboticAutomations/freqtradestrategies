    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
             proposed_leverage: float, max_leverage: float, side: str,
             **kwargs) -> float:
        """
        Customize leverage for each new trade.
    
        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        window_size = 50
        # Obtain historical candle data for the given pair and timeframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    
        # Extract required historical data for indicators
        historical_close_prices = dataframe['close'].tail(window_size)
        historical_high_prices = dataframe['high'].tail(window_size)
        historical_low_prices = dataframe['low'].tail(window_size)
    
        # Set base leverage
        base_leverage = 10
        
        # Calculate RSI and ATR based on historical data using TA-Lib
        rsi_values = ta.RSI(historical_close_prices, timeperiod=14)  # Adjust the time period as needed
        atr_values = ta.ATR(historical_high_prices, historical_low_prices, historical_close_prices, timeperiod=14)  # Adjust the time period as needed
    
        # Calculate MACD and SMA based on historical data
        macd_line, signal_line, _ = ta.MACD(historical_close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        sma_values = ta.SMA(historical_close_prices, timeperiod=20)
    
        # Get the current RSI and ATR values from the last data point in the historical window
        current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50.0  # Default value if no data available
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0.0  # Default value if no data available
    
        # Get the current MACD and SMA values
        current_macd = macd_line[-1] - signal_line[-1] if len(macd_line) > 0 and len(signal_line) > 0 else 0.0
        current_sma = sma_values[-1] if len(sma_values) > 0 else 0.0
        
        # Define dynamic thresholds for RSI and ATR for leverage adjustments
        # Set default values or use non-NaN values if available
        dynamic_rsi_low = np.nanmin(rsi_values) if len(rsi_values) > 0 and not np.isnan(np.nanmin(rsi_values)) else 30.0
        dynamic_rsi_high = np.nanmax(rsi_values) if len(rsi_values) > 0 and not np.isnan(np.nanmax(rsi_values)) else 70.0
        dynamic_atr_low = np.nanmin(atr_values) if len(atr_values) > 0 and not np.isnan(np.nanmin(atr_values)) else 0.002
        dynamic_atr_high = np.nanmax(atr_values) if len(atr_values) > 0 and not np.isnan(np.nanmax(atr_values)) else 0.005
    
        # Print variables for debugging
        print("Historical Close Prices:", historical_close_prices)
        print("RSI Values:", rsi_values)
        print("ATR Values:", atr_values)
        print("Current RSI:", current_rsi)
        print("Current ATR:", current_atr)
        print("Current MACD:", current_macd)
        print("Current SMA:", current_sma)
        print("Dynamic RSI Low:", dynamic_rsi_low)
        print("Dynamic RSI High:", dynamic_rsi_high)
        print("Dynamic ATR Low:", dynamic_atr_low)
        print("Dynamic ATR High:", dynamic_atr_high)
    
        # Leverage adjustment factors
        long_increase_factor = 2.0  # Increase leverage to double the base leverage for long positions
        long_decrease_factor = 0.5  # Decrease leverage to half the base leverage for long positions
        short_increase_factor = 2.0  # Increase leverage to double the base leverage for short positions
        short_decrease_factor = 0.5  # Decrease leverage to half the base leverage for short positions
        volatility_decrease_factor = 0.8  # Decrease leverage to 80% of the base leverage when volatility is high
    
        # Adjust leverage for long trades
        if side == 'long':
             # Adjust leverage for short trades based on dynamic thresholds and current RSI
            if current_rsi < dynamic_rsi_low:
                base_leverage *= long_increase_factor
            elif current_rsi > dynamic_rsi_high:
                base_leverage *= long_decrease_factor
    
            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor
    
            # Adjust leverage based on MACD and SMA
            if current_macd > 0:
                base_leverage *= long_increase_factor
            if current_rate < current_sma:
                base_leverage *= long_decrease_factor
    
        # Adjust leverage for short trades
        elif side == 'short':
             # Adjust leverage for short trades based on dynamic thresholds and current RSI
            if current_rsi > dynamic_rsi_high:
                base_leverage *= short_increase_factor
            elif current_rsi < dynamic_rsi_low:
                base_leverage *= short_decrease_factor
    
            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor
    
            # Adjust leverage based on MACD and SMA
            if current_macd < 0:
                base_leverage *= short_increase_factor  # Increase leverage for potential downward movement
            if current_rate > current_sma:
                base_leverage *= short_decrease_factor  # Decrease leverage if price is above the moving average
        
        else:
            return proposed_leverage  # Return the proposed leverage if side is neither 'long' nor 'short'
    
        # Apply maximum and minimum limits to the adjusted leverage
        adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)  # Apply max and min limits
    
        # Print variables for debugging
        print("Proposed Leverage:", proposed_leverage)
        print("Adjusted Leverage:", adjusted_leverage)
    
        return adjusted_leverage  # Return the adjusted leverage