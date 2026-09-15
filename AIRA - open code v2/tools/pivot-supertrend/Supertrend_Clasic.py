def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr_len: int, mult: float):
        """
        SuperTrend clasic indicator
        return supertrend_line direction +1 bull  -1 bear
        """
        atr_len = int(max(2, atr_len))
        tr = GPTStrategyV6._true_range(high, low, close)
        atr = GPTStrategyV6._rma(tr, atr_len)

        hl2 = (high + low) / 2.0
        upperband = hl2 + (mult * atr)
        lowerband = hl2 - (mult * atr)

        fu = upperband.copy()
        fl = lowerband.copy()

        st = pd.Series(index=close.index, dtype="float64")
        direction = pd.Series(index=close.index, dtype="float64")

        # start  ATR
        first_valid = atr.first_valid_index()
        if first_valid is None:
            return st, direction

        idx = close.index.get_loc(first_valid)

        # Initialize bands up to idx
        for i in range(idx + 1, len(close)):
            prev_i = i - 1

            # final upper
            if (upperband.iat[i] < fu.iat[prev_i]) or (close.iat[prev_i] > fu.iat[prev_i]):
                fu.iat[i] = upperband.iat[i]
            else:
                fu.iat[i] = fu.iat[prev_i]

            # final lower
            if (lowerband.iat[i] > fl.iat[prev_i]) or (close.iat[prev_i] < fl.iat[prev_i]):
                fl.iat[i] = lowerband.iat[i]
            else:
                fl.iat[i] = fl.iat[prev_i]

        #  SuperTrend  idx
        st.iat[idx] = fu.iat[idx] if close.iat[idx] <= fu.iat[idx] else fl.iat[idx]
        direction.iat[idx] = 1.0 if close.iat[idx] > st.iat[idx] else -1.0

        for i in range(idx + 1, len(close)):
            prev_i = i - 1
            prev_st = st.iat[prev_i]

            if np.isnan(prev_st):
                st.iat[i] = fu.iat[i] if close.iat[i] <= fu.iat[i] else fl.iat[i]
            else:
                #  previous ST used upper band
                if prev_st == fu.iat[prev_i]:
                    st.iat[i] = fu.iat[i] if close.iat[i] <= fu.iat[i] else fl.iat[i]
                else:
                    st.iat[i] = fl.iat[i] if close.iat[i] >= fl.iat[i] else fu.iat[i]

            direction.iat[i] = 1.0 if close.iat[i] > st.iat[i] else -1.0

        return st, direction