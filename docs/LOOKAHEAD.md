# Lookahead bias

A strategy has *lookahead bias* when it reads information that would not have
existed at the moment it claims to trade. Backtests of such a strategy are
meaningless: they are scored against data the strategy has already seen.

Freqtrade ships `freqtrade lookahead-analysis`, which detects this empirically by
running the strategy. **That was deliberately not used here.** Running 1,286
unverified files downloaded from a public channel is exactly the thing this
repository warns you not to do. Instead every file was parsed with Python's `ast`
module and matched against the patterns below — no code was executed.

Static detection is a trade-off: it finds patterns, not proof. A 🔴 file is very
likely biased; a 🟢 file is *unflagged*, not *verified*. If you intend to trade a
strategy from here, run Freqtrade's own analysis on it in a sandbox.

## Rules

| Rule | Severity | Why it leaks | Hits |
|---|---|---|---:|
| `backfill` | **high** | bfill()/fillna(method='bfill') propagates later values into earlier rows | 23 |
| `extrema_scan` | **high** | argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet | 52 |
| `future_name` | **high** | a column literally named future/target/label/lookahead is a forecast horizon, not a live signal | 0 |
| `interpolate_back` | **high** | interpolate() with a backward/both direction fills from later rows | 0 |
| `negative_diff` | **high** | diff()/pct_change() with a negative period compares against the future | 0 |
| `negative_shift` | **high** | shift() with a negative period pulls a future candle backwards | 34 |
| `reverse_index` | **high** | the price dataframe itself is reversed, which turns any subsequent backward-looking op into a forward-looking one | 0 |
| `rolling_center` | **high** | rolling(center=True) centres the window, so half of it is future data | 59 |
| `describe_stats` | medium | describe()/mean() over the full column is computed once from all data | 1 |
| `global_scaler` | medium | a scaler/encoder fitted on the whole dataframe leaks test-set statistics into training | 2 |
| `iloc_last_assign` | medium | assigning df.iloc[-1] (or [-1]) across a column broadcasts the newest value backwards | 34 |
| `whole_series_stat` | medium | max()/min()/quantile()/rank() over the entire series uses the full backtest range at every candle | 108 |

Only **high** findings move a file into `dirty LA/`. Medium findings are reported
in [CATALOG.md](../CATALOG.md) as 🟡 and left in place, because each has a
legitimate use that is indistinguishable from the buggy one without reading the
surrounding logic — `dataframe['close'].max()` is wrong in an indicator and fine
in a logging line.

Two rules were tightened after review: `[::-1]` now only fires on the price
dataframe itself (it was matching convolution-kernel flips such as
`weights[::-1]`), and `argrelextrema` must be *called* rather than merely
imported. That reduced the flagged set from 310 to 137.

## Flagged strategies (137)


### [`ARIMA.py`](dirty%20LA/from-AIRA/arima-statistical/ARIMA.py)

*Strategy built on RSI, EMA, SMA, ATR.*

- **`extrema_scan`** (3×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 447
  def find_peaks(self, x, height=None, threshold=None, distance=None,
  ```
  ```python
  # line 872
  high_peaks, high_properties = peak_finder.find_peaks(dataframe['OHLC4'].values, height=None, threshold=None,
  ```

### [`ARIMA_FUTURES.py`](dirty%20LA/from-AIRA/arima-statistical/ARIMA_FUTURES.py)

*Strategy built on RSI, EMA, SMA, ATR.*

- **`extrema_scan`** (3×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 436
  def find_peaks(self, x, height=None, threshold=None, distance=None,
  ```
  ```python
  # line 855
  high_peaks, high_properties = peak_finder.find_peaks(dataframe['OHLC4'].values, height=None, threshold=None,
  ```

### [`AlexBattleTankKillerV4.py`](dirty%20LA/from-AIRA/battle-tank/AlexBattleTankKillerV4.py)

*Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 665
  series = series.interpolate(method='linear').bfill().ffill()  # FIXED
  ```

### [`AlexBattleTankKillerV48.py`](dirty%20LA/from-AIRA/battle-tank/AlexBattleTankKillerV48.py)

*Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 739
  series = series.interpolate(method='linear', limit_direction='both').ffill().bfill()
  ```

### [`NOTankAi_15.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_15.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 704
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 705
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NOTankAi_15_Cleaned.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_15_Cleaned.py)

*Strategy built on RSI, MACD, SMA, ATR.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 332
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 333
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NOTankAi_15__2.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_15__2.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 547
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 548
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NOTankAi_15__3.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_15__3.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 547
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 548
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NOTankAi_15m.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_15m.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 568
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 569
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NOTankAi_5.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_5.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 551
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 552
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NOTankAi_Futures.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_Futures.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 543
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 544
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NOTankAi_vX.py`](dirty%20LA/from-AIRA/battle-tank/NOTankAi_vX.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 544
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 545
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NoTankAi_19_2.py`](dirty%20LA/from-AIRA/battle-tank/NoTankAi_19_2.py)

*NOTankAi_19_2 — LOOKAHEAD-BIAS FIX on top of v19_1.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 852
  max_peaks = argrelextrema(dataframe["close"].values, np.greater, order=extrema_order)[0]
  ```
  ```python
  # line 853
  min_peaks = argrelextrema(dataframe["close"].values, np.less,    order=extrema_order)[0]
  ```

### [`Tank1Modulus.py`](dirty%20LA/from-AIRA/battle-tank/Tank1Modulus.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 443
  min_peaks = argrelextrema(dataframe["low"].values, np.less, order=order)
  ```
  ```python
  # line 444
  max_peaks = argrelextrema(dataframe["high"].values, np.greater, order=order)
  ```

### [`Tank5HurstDCAV1.py`](dirty%20LA/from-AIRA/battle-tank/Tank5HurstDCAV1.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 1000
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`Tank5HurstDCAV3.py`](dirty%20LA/from-AIRA/battle-tank/Tank5HurstDCAV3.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 998
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`Tank5ModulusDCA.py`](dirty%20LA/from-AIRA/battle-tank/Tank5ModulusDCA.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 442
  min_peaks = argrelextrema(dataframe["ha_close"].values, np.less, order=order)
  ```
  ```python
  # line 443
  max_peaks = argrelextrema(dataframe["ha_close"].values, np.greater, order=order)
  ```

### [`TankAi15.py`](dirty%20LA/from-AIRA/battle-tank/TankAi15.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 430
  min_peaks = argrelextrema(
  ```
  ```python
  # line 434
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 445
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`TankAi15Modulus.py`](dirty%20LA/from-AIRA/battle-tank/TankAi15Modulus.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 599
  min_peaks = argrelextrema(
  ```
  ```python
  # line 603
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 614
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`TankAi15Modulus__2.py`](dirty%20LA/from-AIRA/battle-tank/TankAi15Modulus__2.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 681
  min_peaks = argrelextrema(dataframe["low"].values, np.less, order=order)
  ```
  ```python
  # line 682
  max_peaks = argrelextrema(dataframe["high"].values, np.greater, order=order)
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 693
  .rolling(window=5, win_type="gaussian", center=True)
  ```

### [`TankAi15_Futures.py`](dirty%20LA/from-AIRA/battle-tank/TankAi15_Futures.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 540
  min_peaks = argrelextrema(
  ```
  ```python
  # line 544
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 555
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`TankAi15_WIP.py`](dirty%20LA/from-AIRA/battle-tank/TankAi15_WIP.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 611
  min_peaks = argrelextrema(
  ```
  ```python
  # line 615
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 626
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`TankAi15__2.py`](dirty%20LA/from-AIRA/battle-tank/TankAi15__2.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 515
  min_peaks = argrelextrema(
  ```
  ```python
  # line 519
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 530
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`TankAi15__3.py`](dirty%20LA/from-AIRA/battle-tank/TankAi15__3.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 656
  min_peaks = argrelextrema(
  ```
  ```python
  # line 660
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 671
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`HPStrategyV8UltraDCACSL.py`](dirty%20LA/from-AIRA/channels/HPStrategyV8UltraDCACSL.py)

*Strategy built on RSI, CCI, Donchian.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 169
  (dataframe['recovery'].shift(-1) == True)).astype(int)
  ```
  ```python
  # line 169
  (dataframe['recovery'].shift(-1) == True)).astype(int)
  ```

### [`HPStrategyV8UltraDCACSL__2.py`](dirty%20LA/from-AIRA/channels/HPStrategyV8UltraDCACSL__2.py)

*Strategy built on RSI, CCI, Donchian.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 166
  (dataframe['recovery'].shift(-1) == True)).astype(int)
  ```
  ```python
  # line 166
  (dataframe['recovery'].shift(-1) == True)).astype(int)
  ```

### [`TrainingSignals.py`](dirty%20LA/from-AIRA/cycle-oscillators/TrainingSignals.py)

*Strategy built on RSI, MACD, Bollinger, ADX.*

- **`extrema_scan`** (10×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 446
  p_idx = scipy.signal.argrelextrema(future_df['bb_width'].to_numpy(), np.greater_equal, order=order)[0]
  ```
  ```python
  # line 468
  v_idx = scipy.signal.argrelextrema(future_df['bb_width'].to_numpy(), np.less_equal, order=order)[0]
  ```

### [`DataframePopulator.py`](dirty%20LA/from-AIRA/elliott-wave/DataframePopulator.py)

*Strategy built on RSI, MACD, Bollinger, EMA.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 591
  future_df['future_close'] = future_df[price_col].shift(-lookahead_win)
  ```
  ```python
  # line 630
  future_df['future_dwt'] = future_df['full_dwt'].shift(-lookahead_win)
  ```

### [`GeneTrader_gen5_1735014093_4541.py`](dirty%20LA/from-AIRA/elliott-wave/GeneTrader_gen5_1735014093_4541.py)

*PASTE OUTPUT FROM HYPEROPT HERE.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 539
  sup_series = informative['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`PCA3_short.py`](dirty%20LA/from-AIRA/elliott-wave/PCA3_short.py)

*Strategy built on RSI, MACD, Bollinger, EMA.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 704
  future_df['future_close'] = future_df[price_col].shift(-lookahead)
  ```
  ```python
  # line 737
  future_df['future_dwt'] = future_df['dwt_full'].shift(-lookahead)
  ```
- **`rolling_center`** (2×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 467
  res_series = dataframe['high'].rolling(window=5, center=True).apply(lambda row: is_resistance(row),
  ```
  ```python
  # line 469
  sup_series = dataframe['low'].rolling(window=5, center=True).apply(lambda row: is_support(row),
  ```

### [`legendary_ta.py`](dirty%20LA/from-AIRA/freqai-models/legendary_ta.py)

*FreqAI model built on Stochastic, Fisher, Pivot, FreqAI.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 398
  high_indices = argrelextrema(dataframe['high'].to_numpy(), np.greater)
  ```
  ```python
  # line 399
  low_indices = argrelextrema(dataframe['low'].to_numpy(), np.less)
  ```

### [`GKD_FisherTransformV4_ML_V1_2_RU-4_FIXED.py`](dirty%20LA/from-AIRA/gkd/GKD_FisherTransformV4_ML_V1_2_RU-4_FIXED.py)

*# Enhanced Fisher Transform Strategy with ML/RL Integration - V1.2 BALANCED.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 479
  returns = recent_data['close'].pct_change().shift(-1)
  ```
  ```python
  # line 479
  returns = recent_data['close'].pct_change().shift(-1)
  ```

### [`Peekaboo.py`](dirty%20LA/from-AIRA/heikin-ashi/Peekaboo.py)

*Strategy built on RSI, EMA, SMA, Heikin-Ashi.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 291
  min_peaks = argrelextrema(
  ```
  ```python
  # line 295
  max_peaks = argrelextrema(
  ```

### [`PeekabooV2.py`](dirty%20LA/from-AIRA/heikin-ashi/PeekabooV2.py)

*Strategy built on RSI, EMA, SMA, Heikin-Ashi.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 303
  min_peaks = argrelextrema(
  ```
  ```python
  # line 307
  max_peaks = argrelextrema(
  ```

### [`PeekabooV2__2.py`](dirty%20LA/from-AIRA/heikin-ashi/PeekabooV2__2.py)

*Strategy built on RSI, EMA, SMA, Heikin-Ashi.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 297
  min_peaks = argrelextrema(
  ```
  ```python
  # line 301
  max_peaks = argrelextrema(
  ```

### [`PeekabooV2_updated.py`](dirty%20LA/from-AIRA/heikin-ashi/PeekabooV2_updated.py)

*Strategy built on RSI, EMA, SMA, Heikin-Ashi.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 281
  min_peaks = argrelextrema(
  ```
  ```python
  # line 285
  max_peaks = argrelextrema(
  ```

### [`PeekabooV4.py`](dirty%20LA/from-AIRA/heikin-ashi/PeekabooV4.py)

*Strategy built on RSI, EMA, SMA, ATR.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 334
  high_peaks, high_properties = find_peaks(dataframe['high'].values, prominence=dataframe['high'].values / 100 * ptc_target,
  ```
  ```python
  # line 337
  lower_peaks, low_properties = find_peaks(-dataframe['low'].values, prominence=dataframe['low'].values / 100 * ptc_target,
  ```

### [`PeekabooV4002.py`](dirty%20LA/from-AIRA/heikin-ashi/PeekabooV4002.py)

*Strategy built on RSI, EMA, SMA, ATR.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 334
  high_peaks, high_properties = find_peaks(dataframe['high'].values, prominence=dataframe['high'].values / 300 * ptc_target,
  ```
  ```python
  # line 337
  lower_peaks, low_properties = find_peaks(-dataframe['low'].values, prominence=dataframe['low'].values / 300 * ptc_target,
  ```

### [`Peekaboo__2.py`](dirty%20LA/from-AIRA/heikin-ashi/Peekaboo__2.py)

*Strategy built on RSI, EMA, SMA, Heikin-Ashi.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 306
  min_peaks = argrelextrema(
  ```
  ```python
  # line 310
  max_peaks = argrelextrema(
  ```

### [`DarkProphetRL.py`](dirty%20LA/from-AIRA/hurst-cycle/DarkProphetRL.py)

*$$$$$$$ | ______ ______ $$ | __ $$$$$$$ | ______ ______ ______ ______ ______.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 421
  df.bfill(inplace=True)
  ```

### [`FFTEWO.py`](dirty%20LA/from-AIRA/hurst-cycle/FFTEWO.py)

*Strategy built on RSI, EMA, SMA, ATR.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 528
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`haGradient.py`](dirty%20LA/from-AIRA/hurst-cycle/haGradient.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 542
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`backtest-result-2025-06-17_16-19-01_FTPv5.py`](dirty%20LA/from-AIRA/hyperopt-and-tools/backtest-result-2025-06-17_16-19-01_FTPv5.py)

*Strategy built on RSI, Bollinger, SMA, ADX.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 641
  dataframe['chikou_span'] = close.shift(-26)
  ```
  ```python
  # line 641
  dataframe['chikou_span'] = close.shift(-26)
  ```

### [`FTPv5_20250630_opt_btc_bnb.py`](dirty%20LA/from-AIRA/ichimoku/FTPv5_20250630_opt_btc_bnb.py)

*Strategy built on RSI, Bollinger, SMA, ADX.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 739
  dataframe['chikou_span'] = close.shift(-26)
  ```
  ```python
  # line 739
  dataframe['chikou_span'] = close.shift(-26)
  ```

### [`HarmonicDivergencev3.py`](dirty%20LA/from-AIRA/ichimoku/HarmonicDivergencev3.py)

*FreqAI model built on RSI, MACD, Bollinger, EMA.*

- **`negative_shift`** (1×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 1078
  dataframe['chikou_span'] = dataframe['close'].shift(-displacement)
  ```

### [`ALEXGKD_FisherTransformV61_ML.py`](dirty%20LA/from-AIRA/machine-learning/ALEXGKD_FisherTransformV61_ML.py)

*# Enhanced Fisher Transform Strategy with ML/RL Integration - V1.2 BALANCED.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 491
  returns = recent_data['close'].pct_change().shift(-1)
  ```
  ```python
  # line 491
  returns = recent_data['close'].pct_change().shift(-1)
  ```

### [`Alex_GKD_FisherTransformV5_ML.py`](dirty%20LA/from-AIRA/machine-learning/Alex_GKD_FisherTransformV5_ML.py)

*# Enhanced Fisher Transform Strategy with ML/RL Integration - V1.2 BALANCED.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 491
  returns = recent_data['close'].pct_change().shift(-1)
  ```
  ```python
  # line 491
  returns = recent_data['close'].pct_change().shift(-1)
  ```

### [`AutoGluonTS15.py`](dirty%20LA/from-AIRA/machine-learning/AutoGluonTS15.py)

*Strategy built on MACD, Machine Learning.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 239
  df[available_cols] = df[available_cols].ffill().bfill()
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 726
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`AutoGluonTankAIHybrid.py`](dirty%20LA/from-AIRA/machine-learning/AutoGluonTankAIHybrid.py)

*Strategy built on RSI, ATR, Heikin-Ashi, Machine Learning.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 418
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`AutoGluonTankAIHybridV2.py`](dirty%20LA/from-AIRA/machine-learning/AutoGluonTankAIHybridV2.py)

*Strategy built on RSI, ATR, Heikin-Ashi, Machine Learning.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 420
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`Decibel_Mov_-_Future1_beta3.py`](dirty%20LA/from-AIRA/machine-learning/Decibel_Mov_-_Future1_beta3.py)

*Strategy built on Bollinger, SMA, CCI, Machine Learning.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 89
  processed_data['target'] = (processed_data['close'].shift(-1) > processed_data['close']).astype(int)
  ```
  ```python
  # line 89
  processed_data['target'] = (processed_data['close'].shift(-1) > processed_data['close']).astype(int)
  ```

### [`PeekabooCUSUM.py`](dirty%20LA/from-AIRA/machine-learning/PeekabooCUSUM.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 395
  high_peaks, high_properties = find_peaks(dataframe['high'].values, prominence=dataframe['high'].values / 100 * ptc_target,
  ```
  ```python
  # line 398
  lower_peaks, low_properties = find_peaks(-dataframe['low'].values, prominence=dataframe['low'].values / 100 * ptc_target,
  ```

### [`PeekabooV4.py`](dirty%20LA/from-AIRA/machine-learning/PeekabooV4.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 387
  high_peaks, high_properties = find_peaks(dataframe['high'].values, prominence=dataframe['high'].values / 100 * ptc_target,
  ```
  ```python
  # line 390
  lower_peaks, low_properties = find_peaks(-dataframe['low'].values, prominence=dataframe['low'].values / 100 * ptc_target,
  ```

### [`test_wvpredict.py`](dirty%20LA/from-AIRA/machine-learning/test_wvpredict.py)

*Test program for verifying wavelet prediction.*

- **`negative_shift`** (3×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 671
  dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
  ```
  ```python
  # line 701
  dataframe[col] = dataframe[col].shift(-lookahead)
  ```

### [`ANF_v2.py`](dirty%20LA/from-AIRA/murrey-gann/ANF_v2.py)

*EN: ANF's four-layer stack plus an HMM regime controller on top.*

- **`backfill`** (2×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 455
  .mean().bfill().ffill().values
  ```
  ```python
  # line 4666
  series = series.ffill().bfill()
  ```
- **`negative_shift`** (3×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 2036
  forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
  ```
  ```python
  # line 2073
  forward_highs = df['high'].rolling(forward_periods).max().shift(-forward_periods)
  ```

### [`ANF_v2s.py`](dirty%20LA/from-AIRA/murrey-gann/ANF_v2s.py)

*EN: ANF's four-layer stack plus an HMM regime controller on top.*

- **`backfill`** (2×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 473
  .mean().bfill().ffill().values
  ```
  ```python
  # line 4684
  series = series.ffill().bfill()
  ```
- **`negative_shift`** (3×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 2054
  forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
  ```
  ```python
  # line 2091
  forward_highs = df['high'].rolling(forward_periods).max().shift(-forward_periods)
  ```

### [`AlexNexusForgeV7.py`](dirty%20LA/from-AIRA/murrey-gann/AlexNexusForgeV7.py)

*Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.*

- **`negative_shift`** (4×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 372
  (dataframe['high'] > dataframe['high'].shift(-1))
  ```
  ```python
  # line 377
  (dataframe['low'] < dataframe['low'].shift(-1))
  ```

### [`AlexNexusForgeV8AIV2.py`](dirty%20LA/from-AIRA/murrey-gann/AlexNexusForgeV8AIV2.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 227
  .mean().bfill().ffill().values
  ```
- **`negative_shift`** (3×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 998
  forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
  ```
  ```python
  # line 1035
  forward_highs = df['high'].rolling(forward_periods).max().shift(-forward_periods)
  ```
- **`rolling_center`** (2×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 226
  pd.Series(y).rolling(window=3, center=True)
  ```
  ```python
  # line 1533
  smoothed_result = result_series.rolling(window=3, center=True).mean()
  ```

### [`AlexNexusForgeV8AIV8.py`](dirty%20LA/from-AIRA/murrey-gann/AlexNexusForgeV8AIV8.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`backfill`** (2×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 257
  .mean().bfill().ffill().values
  ```
  ```python
  # line 2664
  series = series.ffill().bfill()
  ```
- **`negative_shift`** (3×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 1028
  forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
  ```
  ```python
  # line 1065
  forward_highs = df['high'].rolling(forward_periods).max().shift(-forward_periods)
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 256
  pd.Series(y).rolling(window=3, center=True)
  ```

### [`NoTV15D20241230.py`](dirty%20LA/from-AIRA/murrey-gann/NoTV15D20241230.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 540
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 541
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`BotOwskyStrategy.py`](dirty%20LA/from-AIRA/orderbook-volume/BotOwskyStrategy.py)

*BOT-OWSKY: Advanced Multi-Phase Machine Learning Strategy.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 284
  max_indices = signal.argrelextrema(data, np.greater, order=order)[0]
  ```
  ```python
  # line 290
  min_indices = signal.argrelextrema(data, np.less, order=order)[0]
  ```
- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 581
  future_returns = dataframe['close'].pct_change(3).shift(-3)  # 3-candle forward return
  ```
  ```python
  # line 581
  future_returns = dataframe['close'].pct_change(3).shift(-3)  # 3-candle forward return
  ```

### [`BotOwskyStrategy__2.py`](dirty%20LA/from-AIRA/orderbook-volume/BotOwskyStrategy__2.py)

*BOT-OWSKY: Advanced Multi-Phase Machine Learning Strategy.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 885
  max_indices = signal.argrelextrema(data, np.greater, order=order)[0]
  ```
  ```python
  # line 891
  min_indices = signal.argrelextrema(data, np.less, order=order)[0]
  ```
- **`negative_shift`** (3×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 1165
  future_returns = dataframe['close'].pct_change(3).shift(-3)  # 3-candle forward return
  ```
  ```python
  # line 700
  fut = tmp['close'].pct_change(horizon).shift(-horizon)
  ```

### [`BotOwskyt4c0s.py`](dirty%20LA/from-AIRA/orderbook-volume/BotOwskyt4c0s.py)

*BOT-OWSKY: Advanced Multi-Phase Machine Learning Strategy.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 346
  max_indices = signal.argrelextrema(data, np.greater, order=order)[0]
  ```
  ```python
  # line 352
  min_indices = signal.argrelextrema(data, np.less, order=order)[0]
  ```
- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 812
  future_returns = dataframe['close'].pct_change(3).shift(-3)  # 3-candle forward return
  ```
  ```python
  # line 812
  future_returns = dataframe['close'].pct_change(3).shift(-3)  # 3-candle forward return
  ```

### [`QuickAdapterV3.py`](dirty%20LA/from-AIRA/orderbook-volume/QuickAdapterV3.py)

*The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 221
  min_peaks = argrelextrema(
  ```
  ```python
  # line 225
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 236
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`QuickAdapterV3__2.py`](dirty%20LA/from-AIRA/orderbook-volume/QuickAdapterV3__2.py)

*The following freqtrade strategy is released to sponsors of the non-profit FreqAI open-source project.*

- **`extrema_scan`** (3×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 439
  pivots_indices, _, pivots_directions, _ = zigzag(
  ```
  ```python
  # line 470
  n_minima: int = sp.signal.find_peaks(-dataframe[EXTREMA_COLUMN])[0].size
  ```
- **`rolling_center`** (3×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 1063
  window=odd_window, win_type="triang", center=True
  ```
  ```python
  # line 1065
  "smm": series.rolling(window=odd_window, center=True).median(),
  ```

### [`VVRPStrategyBETA0_1.py`](dirty%20LA/from-AIRA/orderbook-volume/VVRPStrategyBETA0_1.py)

*Volume Weighted Range.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 103
  vwap = (sum_pv / sum_v).ffill().bfill()
  ```

### [`VVRPStrategyBETA0_1__2.py`](dirty%20LA/from-AIRA/orderbook-volume/VVRPStrategyBETA0_1__2.py)

*Volume Weighted Range.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 111
  vwap = (sum_pv / sum_v).ffill().bfill()
  ```

### [`external_indicators_v11.py`](dirty%20LA/from-AIRA/orderbook-volume/external_indicators_v11.py)

*Strategy built on EMA, SMA, VWAP.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 332
  return scaled_result.bfill()  # Utilise bfill() au lieu de fillna(method="bfill")
  ```
- **`extrema_scan`** (1×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 127
  peaks,_ = signal.find_peaks(kdy)
  ```

### [`falconTrader.py`](dirty%20LA/from-AIRA/orderbook-volume/falconTrader.py)

*PASTE OUTPUT FROM HYPEROPT HERE.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 476
  sup_series = informative['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`AlexBandSniperV4.py`](dirty%20LA/from-AIRA/pivot-supertrend/AlexBandSniperV4.py)

*Alex BandSniper on 15m Timeframe - OPTIMIZED VERSION.*

- **`backfill`** (4×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 269
  informative_1h = informative_1h.bfill().ffill()
  ```
  ```python
  # line 325
  informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
  ```

### [`AlexBandSniperV4Optimize.py`](dirty%20LA/from-AIRA/pivot-supertrend/AlexBandSniperV4Optimize.py)

*Alex BandSniper on 15m Timeframe - OPTIMIZED VERSION.*

- **`backfill`** (4×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 264
  informative_1h = informative_1h.bfill().ffill()
  ```
  ```python
  # line 320
  informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
  ```

### [`AlexBandSniperV4mod_5m_test.py`](dirty%20LA/from-AIRA/pivot-supertrend/AlexBandSniperV4mod_5m_test.py)

*Strategy built on RSI, MACD, Bollinger, EMA.*

- **`backfill`** (4×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 271
  informative_1h = informative_1h.bfill().ffill()
  ```
  ```python
  # line 327
  informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
  ```

### [`AlexBandSniperV6553.py`](dirty%20LA/from-AIRA/pivot-supertrend/AlexBandSniperV6553.py)

*╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗.*

- **`backfill`** (4×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 322
  informative_1h = informative_1h.bfill().ffill()
  ```
  ```python
  # line 374
  informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
  ```
- **`rolling_center`** (2×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 1402
  center_extr = s_price.rolling(2 * swing + 1, center=True,
  ```
  ```python
  # line 1408
  center_extr = s_price.rolling(2 * swing + 1, center=True,
  ```

### [`haFbmVVRP.py`](dirty%20LA/from-AIRA/pivot-supertrend/haFbmVVRP.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 816
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`haFbmVVRPlev.py`](dirty%20LA/from-AIRA/pivot-supertrend/haFbmVVRPlev.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 825
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`RSI_Futures.py`](dirty%20LA/from-AIRA/rsi-momentum/RSI_Futures.py)

*RSI_Futures — 震荡(range) + 趋势(trend)，独立平仓 + 状态锁（同 K 禁止第二笔不同模式）。.*

- **`backfill`** (5×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 650
  trend_th = trend_th.bfill().ffill()
  ```
  ```python
  # line 651
  range_th2 = range_th2.bfill().ffill()
  ```

### [`StarMark5mv5_1.py`](dirty%20LA/from-AIRA/rsi-momentum/StarMark5mv5_1.py)

*StarMark5mv5_1 strategy.*

- **`backfill`** (2×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 141
  dataframe[f'rsi_{tf}'] = dataframe[f'rsi_{tf}'].bfill() # 首先向后填充
  ```
  ```python
  # line 175
  dataframe[f'adx_{tf}'] = dataframe[f'adx_{tf}'].bfill() # 首先向后填充
  ```

### [`el_extrema.py`](dirty%20LA/from-AIRA/rsi-momentum/el_extrema.py)

*Strategy built on RSI, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 148
  dataframe.loc[argrelextrema(dataframe['close'].values, numpy.less, order=5)[0], '&s-extrema'] = -1
  ```
  ```python
  # line 149
  dataframe.loc[argrelextrema(dataframe['close'].values, numpy.greater, order=5)[0], '&s-extrema'] = 1
  ```

### [`el_extrema_rsiqui.py`](dirty%20LA/from-AIRA/rsi-momentum/el_extrema_rsiqui.py)

*Strategy built on RSI, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 153
  min_extrema_idx = argrelextrema(dataframe['close'].values, numpy.less, order=10)[0]
  ```
  ```python
  # line 154
  max_extrema_idx = argrelextrema(dataframe['close'].values, numpy.greater, order=10)[0]
  ```

### [`picasso_CE_CTI_STC_EMA_1h_V5_4x_3mt_Jan16_np_Jan20.py`](dirty%20LA/from-AIRA/rsi-momentum/picasso_CE_CTI_STC_EMA_1h_V5_4x_3mt_Jan16_np_Jan20.py)

*Strategy built on RSI, MACD, EMA, ADX.*

- **`negative_shift`** (8×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 607
  (dataframe['close'].shift(-1) <= dataframe['ce_chandelier_exit_tp_short']) &
  ```
  ```python
  # line 612
  (dataframe['close'].shift(-1) >= dataframe['ce_chandelier_exit_sl_short']) &
  ```

### [`snd.py`](dirty%20LA/from-AIRA/rsi-momentum/snd.py)

*Strategy built on RSI.*

- **`negative_shift`** (24×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 41
  df["mask_double_top"] = (df['high_roll_max'] >= df['high'].shift(1)) & (df['high_roll_max'] >= df['high'].shift(-1)) & (df['high'] < df['high'].shift(1)) & (df[
  ```
  ```python
  # line 41
  df["mask_double_top"] = (df['high_roll_max'] >= df['high'].shift(1)) & (df['high_roll_max'] >= df['high'].shift(-1)) & (df['high'] < df['high'].shift(1)) & (df[
  ```

### [`haFbm.py`](dirty%20LA/from-AIRA/smart-money-concepts/haFbm.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 786
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`haFbm__2.py`](dirty%20LA/from-AIRA/smart-money-concepts/haFbm__2.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 796
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`BinanceDataSetCreator.py`](dirty%20LA/from-AIRA/uncategorized/BinanceDataSetCreator.py)

*Freqtrade strategy; no docstring and no recognised indicator library.*

- **`negative_shift`** (4×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 487
  df[pRA_plus] = df[pRA].shift(-windowSize)
  ```
  ```python
  # line 495
  df[vRA_plus] = df[vRA].shift(-windowSize)
  ```

### [`test_df.py`](dirty%20LA/from-AIRA/uncategorized/test_df.py)

*Freqtrade strategy; no docstring and no recognised indicator library.*

- **`negative_shift`** (1×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 160
  dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
  ```

### [`test_forecasters.py`](dirty%20LA/from-AIRA/uncategorized/test_forecasters.py)

*Freqtrade strategy; no docstring and no recognised indicator library.*

- **`negative_shift`** (1×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 181
  dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
  ```

### [`test_river.py`](dirty%20LA/from-AIRA/uncategorized/test_river.py)

*Freqtrade strategy; no docstring and no recognised indicator library.*

- **`negative_shift`** (1×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 52
  dataframe['gain_shifted'] = dataframe['gain'].shift(-lookahead)
  ```

### [`AstroQAXGV3_1.py`](dirty%20LA/from-AIRA/wavetrend/AstroQAXGV3_1.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 443
  min_peaks = argrelextrema(
  ```
  ```python
  # line 447
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 458
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`AstroQAXGV4_541.py`](dirty%20LA/from-AIRA/wavetrend/AstroQAXGV4_541.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 533
  min_peaks = argrelextrema(
  ```
  ```python
  # line 537
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 548
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`AlexBTK_CT.py`](dirty%20LA/from-repo/battle-tank/AlexBTK_CT.py)

*Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 318
  series = series.interpolate(method='linear').bfill().ffill()  # FIXED
  ```

### [`AlexBattleTankKiller.py`](dirty%20LA/from-repo/battle-tank/AlexBattleTankKiller.py)

*Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 284
  series = series.interpolate(method='linear').bfill().ffill()  # FIXED
  ```

### [`AlexBattleTankKillerV3.py`](dirty%20LA/from-repo/battle-tank/AlexBattleTankKillerV3.py)

*Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 666
  series = series.interpolate(method='linear').bfill().ffill()  # FIXED
  ```

### [`AlexBattleTankKillerV4H.py`](dirty%20LA/from-repo/battle-tank/AlexBattleTankKillerV4H.py)

*Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 780
  series = series.interpolate(method='linear', limit_direction='both').ffill().bfill()
  ```

### [`NoTankAi_17.py`](dirty%20LA/from-repo/battle-tank/NoTankAi_17.py)

*NOTankAi_17 — Pete Wong's dynamic-RSI variant on top of NOTankAi base.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 747
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 748
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`NoTankAi_19_1.py`](dirty%20LA/from-repo/battle-tank/NoTankAi_19_1.py)

*NOTankAi_19 — Reversal-exit fix + pyramid-up DCA, v18 entry behaviour.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 825
  maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
  ```
  ```python
  # line 826
  minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
  ```

### [`TankAi.py`](dirty%20LA/from-repo/battle-tank/TankAi.py)

*FreqAI model built on RSI, MACD, EMA, SMA.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 420
  min_peaks = argrelextrema(
  ```
  ```python
  # line 424
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (3×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 435
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```
  ```python
  # line 437
  window=10, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`TankAiRevival.py`](dirty%20LA/from-repo/battle-tank/TankAiRevival.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 786
  min_peaks = argrelextrema(
  ```
  ```python
  # line 790
  max_peaks = argrelextrema(
  ```
- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 804
  dataframe['&-s_max'] = dataframe["close"].shift(-order).rolling(
  ```
  ```python
  # line 806
  dataframe['&-s_min'] = dataframe["close"].shift(-order).rolling(
  ```
- **`rolling_center`** (2×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 801
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```
  ```python
  # line 1087
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`kalthetank.py`](dirty%20LA/from-repo/battle-tank/kalthetank.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 817
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`NFIX_BB_RPB.py`](dirty%20LA/from-repo/bb-rpb-tsl/NFIX_BB_RPB.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 6751
  res_series = informative_1d['high'].rolling(window = 5, center=True).apply(lambda row: self.is_resistance(row), raw=True).shift(2)
  ```
  ```python
  # line 6752
  sup_series = informative_1d['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`NFIX_BB_RPB_c7c477d_20211030.py`](dirty%20LA/from-repo/bb-rpb-tsl/NFIX_BB_RPB_c7c477d_20211030.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 3649
  res_series = informative_1d['high'].rolling(window = 5, center=True).apply(lambda row: self.is_resistance(row), raw=True).shift(2)
  ```
  ```python
  # line 3650
  sup_series = informative_1d['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`bbema.py`](dirty%20LA/from-repo/bollinger/bbema.py)

*Default Strategy provided by freqtrade bot.*

- **`negative_shift`** (1×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 85
  dataframe['close10'] = dataframe['close'].shift(periods=-10)
  ```

### [`HEW.py`](dirty%20LA/from-repo/elliott-wave/HEW.py)

*Hurst Elliott Wave (HEW) Strategy for Freqtrade.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 380
  extrema = sorted(list(argrelextrema(dataframe['high'].values, np.greater_equal, order=self.h2//2)[0]) +
  ```
  ```python
  # line 381
  list(argrelextrema(dataframe['low'].values, np.less_equal, order=self.h2//2)[0]))
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 266
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`custom.py`](dirty%20LA/from-repo/elliott-wave/custom.py)

*Strategy built on RSI, EMA, SMA, Elliott Wave.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 639
  copy['valuewhen'] = np.where(copy[condition] > 0, copy[source].shift(-occurrence), 100)
  ```
  ```python
  # line 641
  copy['barrsince'] = copy['colFromIndex'] - copy['colFromIndex'].shift(-occurrence)
  ```

### [`GKD_CT.py`](dirty%20LA/from-repo/gkd/GKD_CT.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 2055
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`HurstCycle3.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycle3.py)

*Strategy built on SMA, ATR, Hurst.*

- **`rolling_center`** (3×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 44
  fld_short_base = dataframe['filtered_close'].rolling(window=self.short_cycle, center=True).mean()
  ```
  ```python
  # line 45
  fld_mid_base = dataframe['filtered_close'].rolling(window=(self.short_cycle * 2), center=True).mean()
  ```

### [`HurstCycle3__2.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycle3__2.py)

*Strategy built on SMA, ATR, Hurst.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 44
  dataframe['fld_short'] = fld_short_base.shift(-half_short)
  ```
  ```python
  # line 45
  dataframe['fld_long'] = fld_long_base.shift(-half_short)
  ```
- **`rolling_center`** (2×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 42
  fld_short_base = dataframe['filtered_close'].rolling(window=self.short_cycle, center=True).mean()
  ```
  ```python
  # line 43
  fld_long_base = dataframe['filtered_close'].rolling(window=self.long_cycle, center=True).mean()
  ```

### [`HurstCycle3__3.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycle3__3.py)

*Strategy built on SMA, ATR, Hurst, Heikin-Ashi.*

- **`rolling_center`** (3×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 71
  fld_short_base = dataframe['filtered_close'].rolling(window=int(self.short_cycle/2), center=True).mean()
  ```
  ```python
  # line 72
  fld_mid_base = dataframe['filtered_close'].rolling(window=self.short_cycle, center=True).mean()
  ```

### [`HurstCycle7.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycle7.py)

*Strategy built on RSI, SMA, Hurst, Heikin-Ashi.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 157
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```
  ```python
  # line 256
  fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
  ```

### [`HurstCycle7__2.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycle7__2.py)

*Strategy built on RSI, SMA, Hurst, Heikin-Ashi.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 164
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```
  ```python
  # line 263
  fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
  ```

### [`HurstCycleV4.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycleV4.py)

*Strategy built on Hurst, Heikin-Ashi.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 124
  fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
  ```
  ```python
  # line 125
  fld_mid_base = dataframe['filtered_close'].rolling(window=self.h0, center=True).mean()
  ```

### [`HurstCycleV5.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycleV5.py)

*Strategy built on RSI, Hurst, Heikin-Ashi.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 212
  fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
  ```
  ```python
  # line 213
  fld_mid_base = dataframe['filtered_close'].rolling(window=self.h0, center=True).mean()
  ```

### [`HurstCycleV5RSI.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycleV5RSI.py)

*Strategy built on RSI, Hurst, Heikin-Ashi, CCI.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 683
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`HurstCycleV5__2.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycleV5__2.py)

*Strategy built on Hurst, Heikin-Ashi.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 110
  fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
  ```
  ```python
  # line 111
  fld_mid_base = dataframe['filtered_close'].rolling(window=self.h0, center=True).mean()
  ```

### [`HurstCycleV6.py`](dirty%20LA/from-repo/hurst-cycle/HurstCycleV6.py)

*Strategy built on Hurst, Heikin-Ashi.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 110
  fld_short_base = dataframe['filtered_close'].rolling(window=self.h2, center=True).mean()
  ```
  ```python
  # line 111
  fld_mid_base = dataframe['filtered_close'].rolling(window=self.h0, center=True).mean()
  ```

### [`KMM.py`](dirty%20LA/from-repo/hurst-cycle/KMM.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 467
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`KitchenSink.py`](dirty%20LA/from-repo/hurst-cycle/KitchenSink.py)

*Strategy built on RSI, EMA, SMA, ATR.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 366
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`TGMA.py`](dirty%20LA/from-repo/hurst-cycle/TGMA.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 390
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`haGradient.py`](dirty%20LA/from-repo/hurst-cycle/haGradient.py)

*______ __ __ __ __ ______ __ __ __ __ __ ______.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 546
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`tsp0chicken.py`](dirty%20LA/from-repo/hurst-cycle/tsp0chicken.py)

*####################################################################################.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 1596
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`tsp0chickenV2.py`](dirty%20LA/from-repo/hurst-cycle/tsp0chickenV2.py)

*####################################################################################.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 1669
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`IchiVwapAdx.py`](dirty%20LA/from-repo/ichimoku/IchiVwapAdx.py)

*IchiVwapAdx Strategy - Multi-Confluence Trend Following with Heiken Ashi Enhancement and Dynamic Stack Size Management.*

- **`backfill`** (1×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 500
  dataframe['chikou_filled'] = dataframe['chikou'].bfill().ffill()
  ```

### [`IchiVwapAdx__2.py`](dirty%20LA/from-repo/ichimoku/IchiVwapAdx__2.py)

*IchiVwapAdx Strategy - Multi-Confluence Trend Following with Heiken Ashi Enhancement and Dynamic Stack Size Management.*

- **`backfill`** (5×) — bfill()/fillna(method='bfill') propagates later values into earlier rows
  ```python
  # line 755
  dataframe['regime_sma_fast'] = ta.SMA(dataframe, timeperiod=self.regime_sma_fast.value).bfill().fillna(dataframe['close'])
  ```
  ```python
  # line 756
  dataframe['regime_sma_slow'] = ta.SMA(dataframe, timeperiod=self.regime_sma_slow.value).bfill().fillna(dataframe['close'])
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 347
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`LorentzianClassification.py`](dirty%20LA/from-repo/machine-learning/LorentzianClassification.py)

*Strategy built on RSI, EMA, SMA, ADX.*

- **`negative_shift`** (4×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 128
  dataframe["y_train"] = np.where(dataframe["close"].shift(-4) < dataframe["close"], -1,
  ```
  ```python
  # line 129
  np.where(dataframe["close"].shift(-4) > dataframe["close"], 1, 0))
  ```

### [`NeuroV1.py`](dirty%20LA/from-repo/machine-learning/NeuroV1.py)

*Strategy built on ATR, Pivot.*

- **`extrema_scan`** (1×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 664
  peaks, _ = scipy.signal.find_peaks(pdf, prominence=prom_min)
  ```

### [`AstroQAV4.py`](dirty%20LA/from-repo/murrey-gann/AstroQAV4.py)

*FreqAI model built on RSI, EMA, SMA, ADX.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 337
  min_peaks = argrelextrema(
  ```
  ```python
  # line 341
  max_peaks = argrelextrema(
  ```
- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 352
  window=5, win_type='gaussian', center=True).mean(std=0.5)
  ```

### [`NostalgiaForInfinityNextGen.py`](dirty%20LA/from-repo/nostalgia-for-infinity/NostalgiaForInfinityNextGen.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 3562
  res_series = informative_1d['high'].rolling(window = 5, center=True).apply(lambda row: self.is_resistance(row), raw=True).shift(2)
  ```
  ```python
  # line 3563
  sup_series = informative_1d['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`NostalgiaForInfinityNextGen_TSL.py`](dirty%20LA/from-repo/nostalgia-for-infinity/NostalgiaForInfinityNextGen_TSL.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`rolling_center`** (2×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 3338
  res_series = informative_1h['high'].rolling(window = 5, center=True).apply(lambda row: self.is_resistance(row), raw=True).shift(2)
  ```
  ```python
  # line 3339
  sup_series = informative_1h['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`NostalgiaForInfinityX.py`](dirty%20LA/from-repo/nostalgia-for-infinity/NostalgiaForInfinityX.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 6915
  res_series = informative_1d['high'].rolling(window = 5, center=True).apply(lambda row: self.is_resistance(row), raw=True).shift(2)
  ```
  ```python
  # line 6916
  sup_series = informative_1d['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`NostalgiaForInfinityXw.py`](dirty%20LA/from-repo/nostalgia-for-infinity/NostalgiaForInfinityXw.py)

*Strategy built on RSI, Bollinger, EMA, SMA.*

- **`rolling_center`** (4×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 8778
  res_series = informative_1d['high'].rolling(window = 5, center=True).apply(lambda row: self.is_resistance(row), raw=True).shift(2)
  ```
  ```python
  # line 8779
  sup_series = informative_1d['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
  ```

### [`MKR.py`](dirty%20LA/from-repo/rsi-momentum/MKR.py)

*Strategy built on RSI, MACD, EMA, SMA.*

- **`negative_shift`** (4×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 111
  dataframe["y_train"] = np.where(dataframe["close"].shift(-4) < dataframe["close"], -1,
  ```
  ```python
  # line 112
  np.where(dataframe["close"].shift(-4) > dataframe["close"], 1, 0))
  ```

### [`RaposaDivergenceV1.py`](dirty%20LA/from-repo/rsi-momentum/RaposaDivergenceV1.py)

*Divergence strategy.*

- **`extrema_scan`** (4×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 162
  low_idx = argrelextrema(data, np.less, order=order)[0]
  ```
  ```python
  # line 188
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  ```

### [`Schism2MM.py`](dirty%20LA/from-repo/schism-hyperion/Schism2MM.py)

*Strategy built on RSI, Order Block.*

- **`extrema_scan`** (1×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 75
  min_peaks = argrelextrema(dataframe['close'].values, np.less, order=100)
  ```

### [`2Candle.py`](dirty%20LA/from-repo/two-candle/2Candle.py)

*Strategy built on RSI, Hurst, Heikin-Ashi.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 652
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`2Candle__2.py`](dirty%20LA/from-repo/two-candle/2Candle__2.py)

*Strategy built on RSI, Hurst, Heikin-Ashi.*

- **`rolling_center`** (1×) — rolling(center=True) centres the window, so half of it is future data
  ```python
  # line 612
  price_data = price_data.rolling(window=window_size, center=True).mean().dropna()
  ```

### [`Minmax.py`](dirty%20LA/from-repo/uncategorized/Minmax.py)

*Freqtrade strategy; no docstring and no recognised indicator library.*

- **`extrema_scan`** (2×) — argrelextrema/find_peaks/zigzag are called to locate turning points, which needs candles that had not closed yet
  ```python
  # line 40
  min_peaks = argrelextrema(slice['close'].values, np.less, order=lookback_size)
  ```
  ```python
  # line 41
  max_peaks = argrelextrema(slice['close'].values, np.greater, order=lookback_size)
  ```

### [`WTAI.py`](dirty%20LA/from-repo/wavetrend/WTAI.py)

*Example of a hybrid FreqAI strat, designed to illustrate how a user may employ.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 327
  df['&s-up_or_down'] = np.where(df["close"].shift(-5) >
  ```
  ```python
  # line 327
  df['&s-up_or_down'] = np.where(df["close"].shift(-5) >
  ```

### [`WTRSIAI.py`](dirty%20LA/from-repo/wavetrend/WTRSIAI.py)

*Example of a hybrid FreqAI strat, designed to illustrate how a user may employ.*

- **`negative_shift`** (2×) — shift() with a negative period pulls a future candle backwards
  ```python
  # line 327
  df['&s-up_or_down'] = np.where(df["close"].shift(-1) >
  ```
  ```python
  # line 327
  df['&s-up_or_down'] = np.where(df["close"].shift(-1) >
  ```
