    def calculate_total_market_cap_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        FIXED: Uses get_pair_dataframe to always work with informative_pairs.
        """
        logger.debug("🚩 Entering calculate_total_market_cap_trend")  # 1

        # Detect futures mode
        current_pair = metadata['pair']
        is_futures = ':' in current_pair

        if is_futures:
            settlement = current_pair.split(':')[1]
            # Futures weighted pairs
            major_coins = {
                f"BTC/USDT:{settlement}": 0.4,
                f"ETH/USDT:{settlement}": 0.2,
                f"BNB/USDT:{settlement}": 0.1,
                f"SOL/USDT:{settlement}": 0.05,
                f"ADA/USDT:{settlement}": 0.05,
            }
        else:
            # Spot pairs
            major_coins = {
                "BTC/USDT": 0.4,
                "ETH/USDT": 0.2,
                "BNB/USDT": 0.1,
                "SOL/USDT": 0.05,
                "ADA/USDT": 0.05,
            }
        logger.debug(f"🟢 major_coins defined: {list(major_coins.keys())}")  #2

        weighted_trend = 0
        total_weight = 0

        # ---- MAIN CHANGE: Use get_pair_dataframe to avoid issues with informative_pairs ----
        for coin_pair, weight in major_coins.items():
            logger.debug(f"🔵 Checking {coin_pair} for market cap trend")  # 3

            # --- NEW: Check that pair/timeframe is in informative_pairs ---
            if (coin_pair, self.timeframe) not in self.informative_pairs():
                logger.warning(f"⚠️ {coin_pair} [{self.timeframe}] is not in informative_pairs. Make sure to include it.")

            try:
                coin_data = self.dp.get_pair_dataframe(coin_pair, self.timeframe)  # <-- CHANGE HERE
                if coin_data is None or coin_data.empty or len(coin_data) < self.total_mcap_ma_period.value:
                    logger.debug(f"❌ {coin_pair} skipped: empty={coin_data is None or coin_data.empty}, candles={len(coin_data) if coin_data is not None else 0}, needed={self.total_mcap_ma_period.value}")  # 4

                    # --- CHANGE: fallback only if futures ---
                    if is_futures and ':' in coin_pair:
                        spot_pair = coin_pair.split(':')[0]
                        coin_data = self.dp.get_pair_dataframe(spot_pair, self.timeframe)  # <-- Also get_pair_dataframe
                        if coin_data is None or coin_data.empty:
                            continue
                        logger.debug(f"🔄 Fallback to spot for {spot_pair}")

                    else:
                        continue

                # Calculate trend using MA
                ma_period = self.total_mcap_ma_period.value
                current_price = coin_data['close'].iloc[-1]
                ma_value = coin_data['close'].rolling(ma_period).mean().iloc[-1]

                # Validation in case the moving average is nan (not enough candles available)
                if pd.isna(ma_value) or ma_value == 0:
                    logger.debug(f"❌ {coin_pair}: MA({ma_period}) not available (probably due to lack of candles).")
                    continue

                # Trend strength
                trend_strength = (current_price - ma_value) / ma_value
                logger.debug(f"✅ {coin_pair} | weight: {weight} | close: {current_price:.2f} | MA({ma_period}): {ma_value:.2f} | trend_strength: {trend_strength:.3%}")  # 5
                weighted_trend += trend_strength * weight
                total_weight += weight

            except Exception as e:
                logger.debug(f"Could not process {coin_pair} for mcap trend: {e}")
                continue

        logger.debug(f"💵🔝 total_weight: {total_weight} | weighted_trend: {weighted_trend}")

        if total_weight > 0:
            mcap_trend = weighted_trend / total_weight
        else:
            mcap_trend = 0

        # Classify trend
        if mcap_trend > 0.05:
            mcap_status = 'bullish'
        elif mcap_trend < -0.05:
            mcap_status = 'bearish'
        else:
            mcap_status = 'neutral'

        logger.debug(f"💵🔝 Market Cap Trend: {mcap_trend:.2%} [{mcap_status.upper()}] [BTC, ETH, BNB, SOL, ADA weighted]")

        dataframe['mcap_trend'] = mcap_trend
        dataframe['mcap_status'] = mcap_status

        return dataframe