from models.PyCryptoBot import PyCryptoBot
from models.PyCryptoBot import truncate as _truncate
from models.helper.LogHelper import Logger
from models.AppState import AppState

class Strategy_CS:
    def __init__(self, app: PyCryptoBot = None, state: AppState = AppState) -> None:
        self.app = app
        self.state = state

        if state.pandas_ta_enabled is False:
            raise ImportError("This Custom Strategy requires pandas_ta, but pandas_ta module is not loaded. Are requirements-advanced.txt modules installed?")

    def tradeSignals(self, data, _df):

        """ 
        #############################################################################################
        If customizing this file it is recommended to make a copy of and name it Strategy_myCS.py
        It will be loaded automatically if pandas-ta is enabled in configuration and it will not
        be overwritten by future updates.
        #############################################################################################
        """

        # buy indicators - using non-traditional settings
        # *** currently requires pandas-ta module and optional talib 

        # will output indicator values in log and telgram when True
        debug = True

        # max possible points - if sell_trigger_override setting is True, this value is used
        self.max_pts = 7
        ### Market Identifier now sets points ###
        # total points required to buy
        self.pts_to_buy = 0 # more points requires more signals to activate and less risky
        # total points to trigger immediate buy pcnt if configured, if not, this is ignored
        self.immed_buy_pts = 0

        # total points required to sell n
        self.pts_to_sell = 0 # only requiring a couple pts will get sell signal quicker
        # total points to trigger immediate sell pcnt if configured, if not, this is ignored
        self.immed_sell_pts = 0

        # Required signals.
        # Specify how many have to be triggered
        # Buys - currently RSI, MACDL, WILLIAMSR - add self.pts_sig_required_buy += 1 to each one
        self.sig_required_buy =  0
        # Sells - currently RSI, MACDL self.pts_sig_required_sell += 1
        self.sig_required_sell = 0 # set to 0 for default
        
        # start variables at 0
        self.buy_pts = 0
        self.sell_pts = 0
        self.totalbuy_pts = 0
        self.totalsell_pts = 0
        self.pts_sig_required_buy = 0
        self.pts_sig_required_sell = 0
        self.market_pts = 0   #defaults to bear strategy on startup
        self.macdl_upper_limit = (data['sma5'].values[0] * 0.005)
        self.macdl_lower_limit = (data['sma5'].values[0] * -0.001)

        # Identify Long and Short term trends with SMA 5,10,20,50,100
        # Use Market Identification to adjust strategies based on risk

        ### Low risk (Bull) ###
        if (
            (data['sma5'].values[0] > data['sma10'].values[0]
            or data['sma5'].values[0] > data['sma15'].values[0])
            and data['sma5'].values[0] > data['sma25'].values[0]
            and data['sma5'].values[0] < data['sma40'].values[0]
            and data['sma25_pc'].values[0] > 0
            and data['sma40_pc'].values[0] > 0
            and data['sma65_pc'].values[0] > 0):

            self.market = "Bull Run - Prepare for liftoff!"
            #set variables
            self.pts_to_buy = 4 
            self.immed_buy_pts = 5 #ALL BULL buys are Immeadiate!
            self.sig_required_buy = 3
            self.pts_to_sell = 7 
            self.immed_sell_pts = 7
            self.sig_required_sell = 4 

        ### High Risk (Bear)
        elif (
            (data['sma10'].values[0] > data['sma5'].values[0]
            or data['sma25'].values[0] > data['sma5'].values[0]) 
            and data['sma40'].values[0] > data['sma5'].values[0]
            and data['sma25_pc'].values[0] < 0):
        
            self.market = "Dump Incoming!!! SELL! SELL! SELL!"
            #set variables
            self.pts_to_buy = 6 # NEVER BUY!!!
            self.immed_buy_pts = 6 # NEVER BUY!!!
            self.sig_required_buy = 4
            self.pts_to_sell = 5
            self.immed_sell_pts = 5
            self.sig_required_sell = 3 

        ### Bullish
        elif (
            data['sma10'].values[0] < data['sma5'].values[0] 
            and data['sma15'].values[0] < data['sma5'].values[0]
            and (data['sma10_pc'].values[0] > 0 
            or data['sma25_pc'].values[0] > 0)
            and data['sma40_pc'].values[0] > 0):
            
            self.market = "Bullish - Get on the bus people!"
            #set variables
            self.pts_to_buy = 5
            self.immed_buy_pts = 6 
            self.sig_required_buy = 3
            self.pts_to_sell = 6 
            self.immed_sell_pts = 7
            self.sig_required_sell = 4 

        ### WAGMI
        elif (
            data['sma10'].values[0] < data['sma5'].values[0] 
            and data['sma5_pc'].values[0] > 0
            and data['sma10_pc'].values[0] > 0
            and data['sma15_pc'].values[0] > 0):
            
            self.market = "WAGMI - Well, Hopefully....."
            #set variables
            self.pts_to_buy = 5
            self.immed_buy_pts = 6 
            self.sig_required_buy = 4
            self.pts_to_sell = 5 
            self.immed_sell_pts = 6
            self.sig_required_sell = 4 

        else:
            self.market = "Sideways - Get a beer..."
            #set variables
            self.pts_to_buy =  6
            self.immed_buy_pts = 7
            self.sig_required_buy = 4
            self.pts_to_sell = 5 
            self.immed_sell_pts = 6
            self.sig_required_sell = 4 


        # RSI with VWMA, percent RSI is above MA for strength
        if ( # Buy when RSI is increasing and above MA by 3% and 70 > RSI > 15
            data["rsi_ma_pcnt"].values[0] >= 3
            and data['rsima10'].values[0] < data['rsi14'].values[0]
            and data['rsi14'].values[0] > 15
            and data['rsi14'].values[0] < 75
        ):
            self.pts_sig_required_buy += 1
            if ( # Strong when RSI is 5% above MA and 55 > RSI > 15
                data["rsi_ma_pcnt"].values[0] > 5
                and data['rsima10'].values[0] < data['rsi14'].values[0]
                and data['rsi14'].values[0] < 60
                and data['rsi14'].values[0] > 20
                and data["rsi14_pc"].values[0] >= 0
            ):
                self.rsi_action = "strongbuy"
                self.buy_pts += 2
            else: 
                self.rsi_action = "buy"
                self.buy_pts += 1
        elif ( # Sell if RSI if decreasing
            data["rsi_ma_pcnt"].values[0] < 5
        ):
            self.pts_sig_required_sell += 1
            # Strong when RSI is more than 5% below MA
            if (
                data["rsi_ma_pcnt"].values[0] < -10
                and data["rsi14_pc"].values[0] < 0
            ):
                self.rsi_action = "strongsell"
                self.sell_pts += 2
            else:
                self.rsi_action = "sell"
                self.sell_pts += 1
        else:
            self.rsi_action = "wait"

        # MACD Leader signal.....
        # for short trading in pycryptobot, we check that MacdLeader > Macdl_sig and upward trend
        if ( # MACDL above Signal by 40% and MACDL change > 20%
            data["macdlead_pc"].values[0] > 4
            and data["macdl"].values[0] > data['macdl_sig'].values[0]
            
        ):
            self.pts_sig_required_buy += 1
            if ( # Strong when MACDL changes 
                data["macdlead"].values[0] > data['macdl'].values[0]
                and data["macdlead_pc"].values[0] > 6
                and data['macdl'].values[0] > data['macdl_sig'].values[0]
            ):
                self.macdl_action = "strongbuy"
                self.buy_pts += 2
            else:
                self.macdl_action = "buy"
                self.buy_pts += 1
        elif ( # Sell when MACDL Starts decreasing
            data["macdlead_pc"].values[0] < 0
            and data['macdl'].values[0] < data['macdl_sig'].values[0]
        ):
            self.pts_sig_required_sell += 1
            if ( # Strong when MACDL below Signal or MACDL < 0
                data["macdl_sg_diff"].values[0] < 0
                and data['macdl'].values[0] < data['macdl_sig'].values[0]
                and data["macdlead"].values[0] < data['macdl_sig'].values[0]
            ):
                self.macdl_action = "strongsell"
                self.sell_pts += 2
            else:
                self.macdl_action = "sell"
                self.sell_pts += 1
        else:
            self.macdl_action = "wait"

        # WaveTrend
        if ( 
            data['wave_ci'].values[0] > data['wave_trend_1'].values[0]
            and data['wave_trend_1'].values[0] > data['wave_trend_2'].values[0]
            and  data['wave_trend_1_pc'].values[0] > 0
        ):
            self.pts_sig_required_buy += 1
            if ( 
                data['wave_ci'].values[0] > data['wave_trend_1'].values[0]
                and  data['wave_trend_1_pc'].values[0] > 0
            ):
                self.wave_action = "strongbuy"
                self.buy_pts += 2
            else:
                self.wave_action = "buy"
                self.buy_pts += 1
        elif ( 
            data['wave_ci'].values[0] < data['wave_trend_1'].values[0]
            and data['wave_trend_1'].values[0] < data['wave_trend_2'].values[0]
            and  data['wave_trend_1_pc'].values[0] < 0
        ):
            self.pts_sig_required_sell += 1
            if ( 
                data['wave_ci'].values[0] < data['wave_trend_1'].values[0]
                and  data['wave_trend_1_pc'].values[0] < 0
            ):
                self.wave_action = "strongsell"
                self.sell_pts += 2
            else:
                self.wave_action = "sell"
                self.sell_pts += 1
        else:
            self.wave_action = "wait"

        # Heinkin-Ashi Trend
        if ( 
            data['HA_close'].values[0] > data['HA_open'].values[0]
        ):
            self.pts_sig_required_buy += 1
            self.ha_trend = "Up"
            self.buy_pts += 1
        elif (            
            data['HA_close'].values[0] < data['HA_open'].values[0]
        ):
            self.pts_sig_required_sell += 1
            self.ha_trend = "Down"
            self.sell_pts += 1
        else:
            self.ha_trend = "wait"

        # Add total buy pts and subtract from sell pts.
        self.totalbuy_pts = (self.buy_pts - self.sell_pts)
        # Add total sell pts and subtract form buy pts.
        self.totalsell_pts = (self.sell_pts - self.buy_pts)  

        # Add total buy pts and subtract from sell pts.
        self.totalbuy_pts = (self.buy_pts - self.sell_pts)
        # Add total sell pts and subtract form buy pts.
        self.totalsell_pts = (self.sell_pts - self.buy_pts)  

        # if (
        #     self.macdl_lower_limit < data['macdlead'].values[0]
        #     and self.macdl_upper_limit > data['macdlead'].values.[0]
        # ):
        #     self.macdlrange = "True"

        # else:
        #     self.macdlrange = "False"

 
        if debug is True:
            indicatorvalues = (        
        
                "|----------------------------------------|      Ch1ck3nt4c0s Strategy v3.0     |----------------------------------------|"
                "\n"
                f"Market Sentiment: {self.market}"
                "\n"
                f"Market: {data['market'].values[0]} | Candle Range: {_truncate(data['range'].values[0],4)}% "
                "\n"
                f"Heinkin Ashi Candles - Open: {_truncate(data['HA_open'].values[0],6)} | Close: {_truncate(data['HA_close'].values[0],6)} | "
                f"Low: {_truncate(data['HA_low'].values[0],6)} | High: {_truncate(data['HA_high'].values[0],6)}"
                "\n"
                f"Standard Candles - Open: {_truncate(data['open'].values[0],6)} | Close: {_truncate(data['close'].values[0],6)}"
                "\n"
                # RSI
                f"RSI: {_truncate(data['rsi14'].values[0], 2)} | RSIpc: {data['rsi14_pc'].values[0]} |"
                f" MA: {_truncate(data['rsima10'].values[0], 2)} | MAPcnt: {data['rsi_ma_pcnt'].values[0]}%"
                "\n"
                # MACD_Leader
                f"MacdLead: {_truncate(data['macdlead'].values[0],6)} | MacdL: {_truncate(data['macdl'].values[0],6)} |"
                f" MacdlSig: {_truncate(data['macdl_sig'].values[0],6)} | MacdLeadpc: {data['macdlead_pc'].values[0]}% | Diff: {data['macdl_sg_diff'].values[0]}%"
                "\n"
                # Wave Trend
                f"Wave CI: {_truncate(data['wave_ci'].values[0],6)} | Wave 1: {_truncate(data['wave_trend_1'].values[0],6)} | "
                f"Wave 2: {_truncate(data['wave_trend_2'].values[0],6)} | Wave 1 PC: {_truncate(data['wave_trend_1_pc'].values[0],6)}" 
                "\n"
                f"SMA5: {_truncate(data['sma5'].values[0],6)} | SMA10: {_truncate(data['sma10'].values[0],6)} | SMA15: {_truncate(data['sma15'].values[0],6)}"
                f" | SMA25: {_truncate(data['sma25'].values[0],6)} | SMA40: {_truncate(data['sma40'].values[0],6)}| SMA105: {_truncate(data['sma105'].values[0],6)}"
                "\n"
                f"SMA5 PC: {_truncate(data['sma5_pc'].values[0],6)} | SMA10 PC: {_truncate(data['sma10_pc'].values[0],6)} | SMA15 PC: {_truncate(data['sma15_pc'].values[0],6)}"
                f" | SMA25 PC: {_truncate(data['sma25_pc'].values[0],6)} | SMA40 PC: {_truncate(data['sma40_pc'].values[0],6)}| SMA105 PC: {_truncate(data['sma105_pc'].values[0],6)}"
                "\n"
                # Actions
                f"MacdL Action: {self.macdl_action} | RSI Action: {self.rsi_action} | Wave Trend: {self.wave_action} |  Trend: {self.ha_trend}"
                "\n"
                f"BuyPts: {self.buy_pts} | Total BuyPts: {self.totalbuy_pts} of {self.pts_to_buy} | Req-BuyPts: {self.pts_sig_required_buy} of {self.sig_required_buy}"
                "\n"
                f"SellPts: {self.sell_pts} | Total SellPts: {self.totalsell_pts} of {self.pts_to_sell} | Req-SellPts: {self.pts_sig_required_sell} of {self.sig_required_sell}"
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
            self.totalbuy_pts >= self.pts_to_buy
            and self.pts_sig_required_buy >= self.sig_required_buy
        ):
            if (
                self.app.getTrailingBuyImmediatePcnt() is not None
                and self.totalbuy_pts >= self.immed_buy_pts
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
            self.totalsell_pts >= self.pts_to_sell
            and self.pts_sig_required_sell >= self.sig_required_sell
        ):
            if (
                self.app.getTrailingSellImmediatePcnt() is not None
                and self.totalsell_pts >= self.immed_sell_pts
            ):
                self.state.trailing_sell_immediate = True
            else:
                self.state.trailing_sell_immediate = False

            return True
        else:
            return False
