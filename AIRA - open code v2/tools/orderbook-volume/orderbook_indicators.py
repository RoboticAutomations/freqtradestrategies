```python
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe

        # --- Basic safety ---
        for col in ["volume","close","open","high","low","bid","ask","delta","total_trades"]:
            if col not in df.columns:
                df[col] = np.nan
        eps = 1e-9

        # --- Trading magnitudes ---
        # Average trade size and execution intensity
        df["avg_trade_size"] = df["volume"] / (df["total_trades"].replace(0, np.nan))
        df["trade_intensity"] = df["total_trades"] / (df["volume"].replace(0, np.nan))

        # --- Delta & imbalances ---
        df["net_imbalance"] = (df["ask"] - df["bid"]) / (df["ask"] + df["bid"] + eps)  # [-1,1]
        df["delta_norm_vol"] = df["delta"] / (df["volume"] + eps)                      # approx [-1,1]
        df["delta_z"] = (df["delta"] - df["delta"].rolling(50).mean()) / (df["delta"].rolling(50).std() + eps)

        # Classic CVD (Cumulative Volume Delta)
        df["cvd"] = df["delta"].fillna(0).cumsum()
        # CVD normalized by price volatility to compare across assets
        ret = (df["close"] / df["close"].shift(1) - 1)
        df["cvd_voladj"] = (df["delta"].fillna(0) / (ret.rolling(50).std() + eps)).cumsum()

        # --- Price-volume relationship (absorption, effort vs result) ---
        # True Range proxy
        tr = (df["high"] - df["low"]).replace(0, np.nan)
        body = (df["close"] - df["open"])
        wick_top = (df["high"] - df[["close","open"]].max(axis=1))
        wick_bot = (df[["close","open"]].min(axis=1) - df["low"])

        df["range_efficiency"] = (body.abs() / (tr + eps))           # 0..1 (body/range)
        df["wick_top_ratio"] = wick_top / (tr + eps)
        df["wick_bot_ratio"] = wick_bot / (tr + eps)

        # Absorption: very high delta against low price movement
        df["absorption_bearish"] = (
            (df["delta_z"] > 2.0) & (body < 0) & (df["range_efficiency"] < 0.3)
        ).astype(int)
        df["absorption_bullish"] = (
            (df["delta_z"] < -2.0) & (body > 0) & (df["range_efficiency"] < 0.3)
        ).astype(int)

        # Price vs CVD divergences
        df["price_change"] = df["close"] - df["close"].shift(1)
        df["cvd_change"] = df["cvd"] - df["cvd"].shift(1)
        df["divergence_bear"] = ((df["price_change"] > 0) & (df["cvd_change"] < 0)).astype(int)
        df["divergence_bull"] = ((df["price_change"] < 0) & (df["cvd_change"] > 0)).astype(int)

        # --- Order Flow Imbalance (OFI) – practical version with your additions ---
        # Simple proxy: bid/ask volume changes between candles
        d_bid = df["bid"] - df["bid"].shift(1)
        d_ask = df["ask"] - df["ask"].shift(1)
        df["ofi_simple"] = (d_ask - d_bid)                              # >0 buying pressure
        df["ofi_norm"] = df["ofi_simple"] / (df["volume"] + eps)

        # --- Level-based imbalances (if included in "imbalances" or "orderflow") ---
        # If 'imbalances' contains % per level (e.g. list of ask/bid ratios), extract quick stats:
        if "imbalances" in df.columns:
            # Count of levels with extreme imbalance (e.g. >= 3:1)
            def _imbal_count(cell):
                try:
                    # supports dict/list with {price: ratio} or list of ratios
                    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                        return np.nan
                    ratios = []
                    if isinstance(cell, dict):
                        ratios = list(cell.values())
                    elif isinstance(cell, (list, tuple)):
                        ratios = list(cell)
                    else:
                        return np.nan
                    return np.sum(np.array(ratios) >= 3.0)
                except Exception:
                    return np.nan

            df["imbalance_levels_ge3"] = df["imbalances"].apply(_imbal_count)

        # Stacked imbalances (continuous ranges on one side of the book)
        df["has_stacked_bid"] = df.get("stacked_imbalances_bid", pd.Series([None]*len(df))).apply(
            lambda x: int(isinstance(x, (list, tuple)) and len(x) > 0)
        )
        df["has_stacked_ask"] = df.get("stacked_imbalances_ask", pd.Series([None]*len(df))).apply(
            lambda x: int(isinstance(x, (list, tuple)) and len(x) > 0)
        )

        # --- Quick signals (optional) ---
        # "Selling climax": very negative delta, high volume, large bottom wick (possible capitulation)
        vol_q = df["volume"].rolling(50).quantile(0.9)
        df["selling_climax"] = (
            (df["delta_norm_vol"] < -0.6) &
            (df["volume"] > vol_q) &
            (df["wick_bot_ratio"] > 0.4)
        ).astype(int)

        # "Buying climax"
        df["buying_climax"] = (
            (df["delta_norm_vol"] > 0.6) &
            (df["volume"] > vol_q) &
            (df["wick_top_ratio"] > 0.4)
        ).astype(int)

        # --- Cleanup of common NaNs/Infs ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df
```