from datetime import date, datetime, timedelta
from models.AppState import AppState
from models.PyCryptoBot import PyCryptoBot
from models.PyCryptoBot import truncate as _truncate
from models.helper.LogHelper import Logger
from models.TradingAccount import TradingAccount
from models.exchange.Granularity import Granularity

class Strategy_CS:
    def __init__(self, app: PyCryptoBot, state: AppState) -> None:
        self.app = app
        self.state = state
        self.use_adjusted_buy_pts = False # default, leave this here and change below
        self.use_adjusted_sell_pts = False # default, leave this here and change below
        self.sell_override_pts = 100 # set default super high so it doesn't work unless a reasonable number is set below
        self.myCS = True

        if self.state.pandas_ta_enabled is False:
            raise ImportError("This Custom Strategy requires pandas_ta, but pandas_ta module is not loaded. Are requirements-advanced.txt modules installed?")

        if self.state.trading_myPta is True:
            from models.Trading_myPta import TechnicalAnalysis
        else:
            from models.Trading_Pta import TechnicalAnalysis
        self.TA = TechnicalAnalysis

    def tradeSignals(self, data, df, current_sim_date, websocket):

        """ 
        #############################################################################################
        If customizing this file it is recommended to make a copy and name it Strategy_myCS.py
        It will be loaded automatically if pandas-ta is enabled in configuration and it will not
        be overwritten by future updates.
        #############################################################################################
        """

        # buy indicators - using non-traditional settings
        # *** currently requires pandas-ta module and optional talib 

        # will output indicator values in log and after a trade in telgram when True
        debug = True

        # create additional DataFrames to analyze for indicators
        # first option is the short_granularity (5m, 15min, 1h, 6h, 1d, etc.)
        # granularity abbreviations can be found in ./models/exchange/Granularity.py
        # next option is websocket if being used, if omitting and enabled websockets later, error will occur
        # self.df_1d = self.addDataFrame("1d", websocket).copy()

        # if only wanting to know EMAbull like smartswitch checks fore, there are already built in
        # functions that will add the required dataframes and return results.  Just use:
        # EMA1hBull = self.app.is1hEMA1226Bull(current_sim_date, websocket)
        # EMA6hBull = self.app.is6hEMA1226Bull(current_sim_date, websocket)
        try:
            # name and add the dataframe
            df_1h = self.app.getAdditionalDf("1h", websocket).copy()
            # set variable to call technical analysis in Trading_Pta (or myPta)
            ta_1h = self.TA(df_1h)
            # add any individual signals/inicators or addAll()
            ta_1h.enableHACandles()
            ta_1h.addWaveTrend(addPC=True)
            # selling override SMA
            ta_1h.addSMA(5)
            ta_1h.addSMA(15, True)
            ta_1h.addSMA(25)
            ta_1h.addSMA(200, True)
            # retrieve the ta results
            df_1h = ta_1h.getDataFrame()
            # name and create last row reference like main dataframe
            data_1h = self.app.getInterval(df_1h)

            df_6h = self.app.getAdditionalDf("6h", websocket).copy()
            # set variable to call technical analysis in Trading_Pta (or myPta)
            ta_6h = self.TA(df_6h)
            # add any individual signals/inicators or addAll()
            ta_6h.enableHACandles()
            ta_6h.addWaveTrend(addPC=True)
            # selling override SMA
            # retrieve the ta results
            df_6h = ta_6h.getDataFrame()
            # name and create last row reference like main dataframe
            data_6h = self.app.getInterval(df_6h)
            
            # df_15m = self.app.getAdditionalDf("15min", websocket).copy()
            # # set variable to call technical analysis in Trading_Pta (or myPta)
            # ta_15m = self.TA(df_15m)
            # # add any individual signals/inicators or addAll()
            # ta_15m.enableHACandles()
            # ta_15m.addWaveTrend(addPC=True)
            # # retrieve the ta results
            # df_15m = ta_15m.getDataFrame()
            # # name and create last row reference like main dataframe
            # data_15m = self.app.getInterval(df_15m)

        except Exception as err:
            raise Exception(f"Custom Strategy DF Error: {err}")
        '''
        try:
            # repeat for any additional, don't recommend more than 1 or 2 additional, adds overhead and API calls
            df_6h = self.app.getAdditionalDf("6h", websocket).copy()
            ta_6h = self.TA(df_6h, self.app.setTotalPeriods())
            ta_6h.addEMA(5,True)
            ta_6h.addEMA(10,True)
            ta_6h.addSMA(50,True)
            if self.app.getMarket() != "SCRT-USDT" and  self.app.getMarket() != "LUNA-USDT":
                ta_6h.addSMA(50,True)
                ta_6h.addSMA(200,True)
            df_6h = ta_6h.getDataFrame()
            data_6h = self.app.getInterval(df_6h)

            # check ema crossovers (these are not standard period lengths, see comments above)
            EMA1hBull = bool(data_1h['ema5'][0] > data_1h['ema10'][0])
            EMA6hBull = bool(data_6h['ema5'][0] > data_6h['ema10'][0])
            if self.app.getMarket() != "SCRT-USDT" and  self.app.getMarket() != "LUNA-USDT":
                SMA6Bull = bool(data_6h['sma50'][0] > data_6h['sma200'][0] and data_6h['sma50_pc'][0] > 0)
            else:
                SMA6Bull = False
        except Exception as err:
            raise Exception(f"Custom Strategy DF Error: {err}")
        '''

        # create some variables to calculate difference between 2 signals
        # these can be used in evaluations below and are not in the dataframe to help keep it cleaner, make
        # changing/adding easier and we only need diff for last row anyway
        # Usage:  self.calcDiff(firstSignal, secondSignal)
        # a negative value means the first signal is below the second signal
        rsi_ma_diff = self.calcDiff(data['rsi14'][0], data['rsima5'][0]) # RSI and MA
#        di_diff = self.calcDiff(data['+di14'][0], data['-di14'][0]) # ADX di+ and di-
        macd_sg_diff = self.calcDiff(data['macd'][0], data['signal'][0]) # Macd and Signal


        # to disable any indicator used in this file, set the buy and sell pts to 0 or comment the whole indicator out
        # the lines for buy and sell pts.
        # ** Be sure to adjust total counts below.

        # if using smartswitch granularity, recommend lowering each pt total by 1 pt due to the EMA Bull being disabled
#        self.max_pts = 9
#        self.sell_override_pts = 9
        self.max_pts = 1
#        self.sell_override_pts = 7 # this is used if sell_trigger_override setting is True, if activate, TSL and preventloss will be bypassed
        # total points required to buy
        self.pts_to_buy = 1 # more points requires more signals to activate, less risk
        # total points to trigger immediate buy if trailingbuyimmediatepcnt is configured, else ignored
        self.immed_buy_pts = 1

        # use adjusted buy or sell pts? Set to True or False, default is false if not added
        # adjusting buy, will subtract sell_pts from total buy_pts before signaling a buy
        self.use_adjusted_buy_pts = True
        # adjusting sell, will subtract buy_pts from total sell_pts before signaling a sell
        self.use_adjusted_sell_pts = False

        # total points required to sell
        self.pts_to_sell = 1 # requiring fewer pts results in quicker sell signal
        # total points to trigger immediate sell if trailingsellimmediatepcnt is configured, else ignored
        self.immed_sell_pts = 1

        # Required signals.
        # Specify how many have to be triggered
        # Buys - currently 1 for WaveTrend- add self.pts_sig_required_buy += 1 to section for each signal
        self.sig_required_buy = 1
        # Sells - currently 1 for WaveTrend - add self.pts_sig_required_sell += 1 to section for each signal
        self.sig_required_sell = 1 # set to 0 for default

        # don't edit these, need to start at 0
        self.buy_pts = 0
        self.sell_pts = 0
        self.pts_sig_required_buy = 0
        self.pts_sig_required_sell = 0

        # WaveTrend
        if ( #
            data['wave_ci'][0] > data['wave_t1'][0]
            and data['wave_t1_pc'][0] > 0
            and data_1h['wave_ci'][0] > data_1h['wave_t1'][0]
            and data_1h['wave_t1_pc'][0] > 0
            and data_6h['wave_ci'][0] > data_6h['wave_t1'][0]
            and data_6h['wave_t1_pc'][0] > 0

        ):
            self.pts_sig_required_buy += 1
            self.wave_action = "buy"
            self.buy_pts += 1
        elif ( #
            data['wave_ci'][0] < data['wave_t1'][0]
            and data['wave_t1_pc'][0] < 0
            and data_1h['wave_ci'][0] < data_1h['wave_t1'][0]
            and data_1h['wave_t1_pc'][0] < 0
            and data_6h['wave_ci'][0] < data_6h['wave_t1'][0]
            and data_6h['wave_t1_pc'][0] < 0
        ):
            self.pts_sig_required_sell += 1
            self.wave_action = "sell"
            self.sell_pts += 1
        else:
            self.wave_action = "wait"



        # Heinkin Ashi Candle direction all time frames
        if(data['close'][0] > data['open'][0]):
            self.HA_15m = "Up"
        else:
            self.HA_15m = "Down"

        if(data_1h['close'][0] > data_1h['open'][0]):
            self.HA_1h = "Up"
        else:
            self.HA_1h = "Down"

        if(data_6h['close'][0] > data_6h['open'][0]):
            self.HA_6h = "Up"
        else:
            self.HA_6h = "Down"

        # Wave Trends Status
        if(data['wave_ci'][0] > data['wave_t1'][0]):
            self.WT_15m = "Up"
        else:
            self.WT_15m = "Down"

        if(data_1h['wave_ci'][0] > data_1h['wave_t1'][0]):
            self.WT_1h = "Up"
        else:
            self.WT_1h = "Down"

        if(data_6h['wave_ci'][0] > data_6h['wave_t1'][0]):
            self.WT_6h = "Up"
        else:
            self.WT_6h = "Down"


        # adjusted buy pts - subtract any sell pts from buy pts
        if self.use_adjusted_buy_pts is True:
            self.buy_pts = self.buy_pts - self.sell_pts

        # adjusted sell pts - subtract any buy pts from sell pts
        if self.use_adjusted_sell_pts is True:
            self.sell_pts = self.sell_pts - self.buy_pts

        if debug is True:
            indicatorvalues = (
                
                "|----------------------------------------|      Ch1ck3nt4c0s Strategy v3.4    |----------------------------------------|"
                "\n"
                f"Market: {data['market'][0]} | BuyPts: {self.buy_pts}:{self.pts_to_buy} | SellPts: {self.sell_pts}:{self.pts_to_sell} | WaveTrend Action: {self.wave_action}" 
                "\n"
                # f"Filter Status: {self.filter} | SMA2: {_truncate(data_1h['sma2'][0],4)} | "
                # f"Buying Override: {_truncate(data_1h['buy_override'][0],4)} | {_truncate(data_1h['buy_override_pc'][0],4)} | "
                # f"Selling Override: {_truncate(data_1h['sell_override'][0],4)} | {_truncate(data_1h['sell_override_pc'][0],4)}"
                # OHCL
                f"Open: {data['open'][0]} | High: {data['high'][0]} | Low: {data['low'][0]} | Close: {data['close'][0]}"
                # HA candle directions all time frames
                "\n"
                f"15m Trend: {self.HA_15m} | 15m Wave: {self.WT_15m} | 1h Trend: {self.HA_1h} | 1h Wave: {self.WT_1h} | 6h Trend: {self.HA_6h} | 6h Wave: {self.WT_6h}  "
                "\n"
                f"|----------------------------------------            Indicator Values           ----------------------------------------|"
                "\n"
                # Wave Trend
                f"15m WaveT1: {_truncate(data['wave_t1'][0],4)} | 15m WaveT2: {_truncate(data['wave_t2'][0],4)} | 15m Wave Ci: {_truncate(data['wave_ci'][0],4)} | 15m Wave T1 PC: {_truncate(data['wave_t1_pc'][0],4)}"
                "\n"
                f"1h WaveT1: {_truncate(data_1h['wave_t1'][0],4)} | 1h WaveT2: {_truncate(data_1h['wave_t2'][0],4)} | 1h Wave Ci: {_truncate(data_1h['wave_ci'][0],4)} | 1h Wave T1 PC: {_truncate(data_1h['wave_t1_pc'][0],4)}"
                "\n"
                f"6h WaveT1: {_truncate(data_6h['wave_t1'][0],4)} | 6h WaveT2: {_truncate(data_6h['wave_t2'][0],4)} | 6h Wave Ci: {_truncate(data_6h['wave_ci'][0],4)} | 6h Wave T1 PC: {_truncate(data_6h['wave_t1_pc'][0],4)}"
                # SMA
                "\n"
                f"SMA5: {_truncate(data['sma5'][0],4)} | {_truncate(data['sma5_pc'][0],4)} | SMA10: {_truncate(data['sma10'][0],4)} | {_truncate(data['sma10_pc'][0],4)} | "
                f"SMA25: {_truncate(data['sma25'][0],4)} | {_truncate(data['sma25_pc'][0],4)} | SMA100: {_truncate(data['sma100'][0],4)} | {_truncate(data['sma100_pc'][0],4)}"
                "\n"
                f"1 Hour - SMA5: {_truncate(data_1h['sma5'][0],4)} | SMA15: {_truncate(data_1h['sma15'][0],4)} | SMA25: {_truncate(data_1h['sma25'][0],4)} | SMA200: {_truncate(data_1h['sma200'][0],4)}"
                "\n"
                "|----------------------------------------|       one bot to rule them all       |----------------------------------------|"
            )
            Logger.info(indicatorvalues)
        else:
            indicatorvalues = ""

        return indicatorvalues

    def buySignal(self) -> bool:

        # non-Traditional buy signal criteria
        # *** currently requires pandas-ta module and optional talib 
        if (
            self.buy_pts >= self.pts_to_buy
            and self.pts_sig_required_buy >= self.sig_required_buy
        ):
            if (
                self.app.getTrailingBuyImmediatePcnt() is not None
                and self.buy_pts >= self.immed_buy_pts
            ):
                self.state.trailing_buy_immediate = True
            else:
                self.state.trailing_buy_immediate = False

            return True
        else:
            return False

    def sellSignal(self) -> bool:

        # non-Traditional sell signal criteria
        # *** currently requires pandas-ta module and optional talib 
        if (
            self.sell_pts >= self.pts_to_sell
            and self.pts_sig_required_sell >= self.sig_required_sell
        ):
            if (
                self.app.getTrailingSellImmediatePcnt() is not None
                and self.sell_pts >= self.immed_sell_pts
            ):
                self.state.trailing_sell_immediate = True
            else:
                self.state.trailing_sell_immediate = False

            return True
        else:
            return False

    def calcDiff(self, first, second) -> None:

        # used to calculate the difference between to values as a percentage
        # negative result means first value is below second value
        if abs(first) == 0:
            return 0
        return (round((first - second) / abs(first) * 100, 2))

    def checkGtTime(self, coTime, length): # -> bool:

        # used to calculate how long the crossover has been in place
        # currently variables in place for SMA crossovers only
        # self.app.sma5gtsma10time, self.app.sma10gtsma50time, self.app.sma50gtsma100time

        if coTime is not None and ((datetime.combine(date.min,datetime.now().time()) - datetime.combine(date.min, coTime)) > timedelta(minutes=length)):
            return (True,(datetime.combine(date.min,datetime.now().time()) - datetime.combine(date.min, coTime)))
        else:
            return (False,0)


    def setCoTime(self, first, second, coTime):
        if first > second:
            if coTime is None:
                return datetime.now().time()
            else:
                return coTime
        else:
            return None
