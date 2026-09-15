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
        # Running this strategy will require uncommenting and/or add the following to Trading_Pta.py in the section    
        # def addAll(self) -> None:
        #     """Adds analysis to the DataFrame""":
        #
        # self.addWilliamsR(14)
        # self.addSMA(5)
        # self.addSMA(10)
        # self.addSMA(20)
        # if self.total_periods >= 50:
        #    self.addSMA(50)
        # if self.total_periods >= 100:
        #    self.addSMA(100)
        # if self.total_periods >= 200:
        #    self.addSMA(200)


        # buy indicators - using non-traditional settings
        # *** currently requires pandas-ta module and optional talib 

        # will output indicator values in log and telgram when True
        debug = True

        # max possible points - if sell_trigger_override setting is True, this value is used
        self.max_pts = 14
        ### Market Identifier now sets points ###
        # total points required to buy
        self.pts_to_buy = 0 
        # total points to trigger immediate buy pcnt if configured, if not, this is ignored
        self.immed_buy_pts = 0

        # total points required to sell
        self.pts_to_sell = 0 
        # total points to trigger immediate sell pcnt if configured, if not, this is ignored
        self.immed_sell_pts = 0

        # Required signals.
        # Specify how many have to be triggered
        # Buys - currently RSI, ADX, MACDL, HYBR MACD, WILLIAMSR, or ERI- add self.pts_sig_required_buy += 1 to each one
        self.sig_required_buy =  0
        # Sells - currently RSI, MACDL, HYBR MACD, WILLIAMSR, or ERI - self.pts_sig_required_sell += 1
        self.sig_required_sell = 0 # set to 0 for default
        
        # start variables at 0
        self.buy_pts = 0
        self.sell_pts = 0
        self.totalbuy_pts = 0
        self.totalsell_pts = 0
        self.pts_sig_required_buy = 0
        self.pts_sig_required_sell = 0
        self.market_pts = 1   

        # Identify Long and Short term trends with SMA 5,10,20,50,100
        # Use Market Identification to adjust strategies based on risk
        # More buy points requires more signals to activate and less risky
        # Only requiring a couple sell pts will get sell signal quicker
        # More required signals will take longer and stronger trends with indicators to get a buy
        # Please use at your own risk. Trade Safe!
        # self.market_pts is a modifier for some setpoints with 0 (default/bull) settings remaining the same.
        # As the trend goes more bearish, variable thresholds will be multiplied upward on a fibonacci scale 138.2%, 150%, and 161.8%.
        # Since we are not buying in the Bear trend it will not have a multiplier, however crossings will remain as they are.

        ### Low risk (Bull) ###
        if data['sma5'].values[0] > data['sma10'].values[0] and data['sma5'].values[0] > data['sma200'].values[0]:
            self.market = "Bull Run - Prepare for liftoff!"
            #set variables
            self.pts_to_buy = 8 
            self.immed_buy_pts = 9
            self.sig_required_buy = 4
            self.pts_to_sell = 10 
            self.immed_sell_pts = 11
            self.sig_required_sell = 4 

        ### High Risk (Bear)
        elif (
            data['sma20'].values[0] > data['sma5'].values[0] 
            and data['sma50'].values[0] > data['sma5'].values[0]
            and data['sma100'].values[0] > data['sma5'].values[0]):
        
            self.market = "Dump Incoming!!! SELL! SELL! SELL!"
            self.market_pts += 1
            #set variables
            self.pts_to_buy = 15 # NEVER BUY!!!
            self.immed_buy_pts = 15 # NEVER BUY!!!
            self.sig_required_buy = 5
            self.pts_to_sell = 8 
            self.immed_sell_pts = 9
            self.sig_required_sell = 4 

        ### Bearish
        elif (
            data['sma10'].values[0] < data['sma5'].values[0] 
            and data['sma20'].values[0] < data['sma5'].values[0] 
            and data['sma100'].values[0] > data['sma50'].values[0]):
            
            self.market = "Bearish - Meh... This might work"
            self.market_pts += 0.618
            #set variables
            self.pts_to_buy = 13 
            self.immed_buy_pts = 14 
            self.sig_required_buy = 5
            self.pts_to_sell = 8 
            self.immed_sell_pts = 9
            self.sig_required_sell = 4 

        ### Bullish
        elif (
            data['sma10'].values[0] < data['sma5'].values[0] 
            and data['sma20'].values[0] < data['sma5'].values[0] 
            and data['sma100'].values[0] < data['sma50'].values[0]):
            
            self.market = "Bullish - You might make money soon..."
            self.market_pts += 0.382
            #set variables
            self.pts_to_buy = 10 
            self.immed_buy_pts = 11 
            self.sig_required_buy = 5
            self.pts_to_sell = 9 
            self.immed_sell_pts = 10
            self.sig_required_sell = 4 

        else:
            self.market = "Sideways - Get a beer..."
            self.market_pts += 0.5
            #set variables
            self.pts_to_buy =  12
            self.immed_buy_pts = 13
            self.sig_required_buy = 5
            self.pts_to_sell = 8 
            self.immed_sell_pts = 9
            self.sig_required_sell = 4 


        # RSI with VWMA, percent RSI is above MA for strength
        if ( # Buy when RSI is increasing and above MA by 3% and 70 > RSI > 15
            data["rsi_ma_pcnt"].values[0] >= (3 * self.market_pts)
            and data['rsima10'].values[0] < data['rsi14'].values[0]
            and data['rsi14'].values[0] > 15
            and data['rsi14'].values[0] < 70
        ):
            self.pts_sig_required_buy += 1
            if ( # Strong when RSI is 5% above MA and 55 > RSI > 15
                data["rsi_ma_pcnt"].values[0] > (5 * self.market_pts)
                and data['rsima10'].values[0] < data['rsi14'].values[0]
                and data['rsi14'].values[0] < 55
                and data['rsi14'].values[0] > 15
                and data["rsi14_pc"].values[0] >= (6 * self.market_pts)
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

        #edited 3-26
        # ADX with percentage of difference between DI+ & DI- for strength
        if ( # DI+ above DI- and a difference of 2.5% and +DI increasing and 65 > ADX > 20
            data["+di14"].values[0] > data["-di14"].values[0]        
            and data["adx14"].values[0] > 20
            and data["adx14"].values[0] < 65
            and data["di_pcnt"].values[0] > (2.5 * self.market_pts)
            
            
        ):
            self.pts_sig_required_buy += 1
            if ( # Strong if DI difference greater than 5% or DI+ is above and ADX 50 > ADX > 20
                data["adx14"].values[0] > 25
                and data["adx14"].values[0] < 50
                and (data["di_pcnt"].values[0] > (5 * self.market_pts)
                    or  data["+di14"].values[0] > data["adx14"].values[0])
            ):
                self.adx_action = "strongbuy"
                self.buy_pts += 2
            else:
                self.adx_action = "buy"
                self.buy_pts += 1
        elif ( # Sell if +DI decreasing or DI+ is below DI-
#            (data["+di_pc"].values[0] < 0
            data["+di14"].values[0] < data["-di14"].values[0]
        ):
            if ( # Strong if DI difference is below -10% or DI- is above ADX
                (data["di_pcnt"].values[0] < -10
                    or data["-di14"].values[0] > data["adx14"].values[0])
                    # might change DI- below ADX, have to watch charts and values
            ):
                self.adx_action = "strongsell"
                self.sell_pts += 2
            else:
                self.adx_action = "sell"
                self.sell_pts += 1
        else:
            self.adx_action = "wait"

        #edited 3-26
        # MACD signal variation using EMA Oscillator & SMA Signal, also using 8, 21, 5
        # in addition to typical > 0 and crossover indicators
        if ( # buy when MACD is climbing and above Signal by 25% or more
            data["macd_sig_pcnt"].values[0] > (8 * self.market_pts)
            and data["macd_pc"].values[0] > (4 * self.market_pts)
            and data["macd"].values[0] > data['signal'].values[0]
        ):
#            self.pts_sig_required_buy += 1
            if ( # Strong when MACD > 0 and difference > 35%    
                data["macd_sig_pcnt"].values[0] > (10 * self.market_pts)
                and data["macd_pc"].values[0] > (6 * self.market_pts)
                and data["macd"].values[0] > data['signal'].values[0]
            ):
                self.macd_action = "strongbuy"
                self.buy_pts += 2
            else:
                self.macd_action = "buy"
                self.buy_pts += 1
        elif ( # Sell when macd is decreasing
            data["macd_pc"].values[0] < 0
        ):
#            self.pts_sig_required_sell += 1
            if ( # Strong if diff between MACD and SIG is < 0
                data["macd_sig_pcnt"].values[0] < 0
            ):
                self.macd_action = "strongsell"
                self.sell_pts += 2
            else:
                self.macd_action = "sell"
                self.sell_pts += 1
        else:
            self.macd_action = "wait"

        # OBV and SMA5 - when OBV is above its SMA, buy and when below, sell
        if ( # Buy when OBV is 1% above SMA and OBV change percent > 2
            data["obv_sm_diff"].values[0] > (1 * self.market_pts)
            and data['obv_pc'].values[0] >= (2 * self.market_pts)
        ):
#            self.pts_sig_required_buy += 1
            self.obv_action = "buy"
            self.buy_pts += 1
        elif ( # Sell when OBV below SMA or OBV is decreasing
            (data["obv_sm_diff"].values[0] < 0
                or data['obv_pc'].values[0] < 0)
        ):
#            self.pts_sig_required_sell += 1
            self.obv_action = "sell"
            self.sell_pts += 1
        else:
            self.obv_action = "wait"

        # MACD Leader signal.....
        # for short trading in pycryptobot, we check that MacdLeader > Macdl_sig and upward trend
        if ( # MACDL above Signal by 40% and MACDL change > 20%
            data["macdlead_pc"].values[0] > (4 * self.market_pts)
            and data["macdlead"].values[0] > data['macdl_sig'].values[0]
            
        ):
            self.pts_sig_required_buy += 1
            if ( # Strong when MACDL changes 
                data["macdlead"].values[0] > data['macdl'].values[0]
                and data["macdlead_pc"].values[0] > (6 * self.market_pts)
                and data['macdlead'].values[0] > data['macdl_sig'].values[0]
            ):
                self.macdl_action = "strongbuy"
                self.buy_pts += 2
            else:
                self.macdl_action = "buy"
                self.buy_pts += 1
        elif ( # Sell when MACDL Starts decreasing
            data["macdlead_pc"].values[0] < 0
            and data['macdlead'].values[0] < data['macdl'].values[0]
        ):
            self.pts_sig_required_sell += 1
            if ( # Strong when MACDL below Signal or MACDL < 0
                data["macdl_sg_diff"].values[0] < 0
                and data['macdlead'].values[0] < data['macdl'].values[0]
                and data["macdlead"].values[0] < data['macdl_sig'].values[0]
            ):
                self.macdl_action = "strongsell"
                self.sell_pts += 2
            else:
                self.macdl_action = "sell"
                self.sell_pts += 1
        else:
            self.macdl_action = "wait"

        # EMA5/WMA5 crossover signal
#         if ( # EMA above WMA
#             data["ema5"].values[0] >= data["ema5_wma5"].values[0]
#             and data["ema5_pc"].values[0] > 0.01
#         ):
# #            self.pts_sig_required_buy += 1
#             if ( # Strong when EMA_pc > 8
#                 data["ema5_pc"].values[0] > 0.1
#             ):
#                 self.emawma_action = "strongbuy"
#                 self.buy_pts += 2
#             else:
#                 self.emawma_action = "buy"
#                 self.buy_pts += 1
#         elif ( # Sell when EMA < WMA
#             data["ema5_pc"].values[0] < 1
#         ):
#             self.pts_sig_required_sell += 1
#             if data["ema5"].values[0] < data["ema5_wma5"].values[0]:
#                 self.emawma_action = "strongsell"
#                 self.sell_pts += 2
#             else:
#                 self.emawma_action = "sell"
#                 self.sell_pts += 1
#         else:
#             self.emawma_action = "wait"
            
        # MACD Hybrid crossover signal
        if ( # Macd Lead above Macd Signal
            data["macdlead"].values[0] > data['macdl_sig'].values[0]          
        ):
            self.pts_sig_required_buy += 1
            if ( # Strong when macd lead above both macd and macd signal
                data["macdlead"].values[0] > data['macdl_sig'].values[0]
                and data["macdlead"].values[0] > data["macdl"].values[0]
            ):
                self.hybmacd_action = "strongbuy"
                self.buy_pts += 2
            else:
                self.hybmacd_action = "buy"
                self.buy_pts += 1
        elif ( # Sell when Macd LEAD < Macd
            data["macdlead"].values[0] < data["macdl"].values[0]
        ):
            self.pts_sig_required_sell += 1
            if (data["macdlead"].values[0] < data['macdl_sig'].values[0]
                and data["macdlead"].values[0] < data["macdl"].values[0]):
                self.hybmacd_action = "strongsell"
                self.sell_pts += 2
            else:
                self.hybmacd_action = "sell"
                self.sell_pts += 1
        else:
            self.hybmacd_action = "wait"

        # WilliamsR Percent Signal
        if ( # williams r confirmation of uptrends with rsi and macdl with positive change
            data['williamsr14'].values[0] > -100
            and data['williamsr14'].values[0] <= -20 
            and data["rsi_ma_pcnt"].values[0] >= (3 * self.market_pts)
            and data['rsima10'].values[0] < data['rsi14'].values[0]
            and data['rsi14'].values[0] > 15
            and data['rsi14'].values[0] < 70
            and data["macdlead_pc"].values[0] > (5 * self.market_pts)        
        ):
            self.pts_sig_required_buy += 1
            if ( # strong when less than -75
                data['williamsr14'].values[0] > -75
                and data['williamsr14'].values[0] <= -40
                and data["rsi_ma_pcnt"].values[0] > (5 * self.market_pts)
                and data['rsima10'].values[0] < data['rsi14'].values[0]
                and data['rsi14'].values[0] < 55
                and data['rsi14'].values[0] > 15
                and data["rsi14_pc"].values[0] >= 0
                and data["macdlead_pc"].values[0] > (7 * self.market_pts) 
            ):
                self.williamsr_action = "strongbuy"
                self.buy_pts += 2
            else:
                self.williamsr_action = "buy"
                self.buy_pts += 1
        elif ( # Sell when williamsr is above -20 and less than 0
            data['williamsr14'].values[0] < 0
            and data['williamsr14'].values[0] > -20     
        ):
            self.pts_sig_required_sell += 1
            if (data['williamsr14'].values[0] < -5
                and data['williamsr14'].values[0] >= -15 
            ):
                self.williamsr_action = "strongsell"
                self.sell_pts += 2
            else:
                self.williamsr_action = "sell"
                self.sell_pts += 1
        else:
            self.williamsr_action = "wait"

        # Elder Ray Index Bull/Bear Power
        if data['eri_buy'].values[0] == True:
            self.pts_sig_required_buy += 1
            self.buy_pts += 1
            self.eri_action = "buy"
        elif data['eri_sell'].values[0] == True:
            self.pts_sig_required_sell += 1
            self.sell_pts += 1
            self.eri_action = "sell"

        # Add total buy pts and subtract from sell pts.
        self.totalbuy_pts = (self.buy_pts - self.sell_pts)
        # Add total sell pts and subtract form buy pts.
        self.totalsell_pts = (self.sell_pts - self.buy_pts)  

 
        if debug is True:
            indicatorvalues = (        
        
                "|----------------------------------------|      Ch1ck3nt4c0s Strategy v1.8      |----------------------------------------|"
                "\n"
                f"Market Sentiment: {self.market}"
                "\n"
                f"Fibonacci Market Multiplier: {self.market_pts}"
                "\n"
                # RSI
                f"RSI: {_truncate(data['rsi14'].values[0], 2)} | RSIpc: {data['rsi14_pc'].values[0]} |"
                f" MA: {_truncate(data['rsima10'].values[0], 2)} | MAPcnt: {data['rsi_ma_pcnt'].values[0]}%"
                "\n"
                # OBV
                f"OBV: {_truncate(data['obv'].values[0], 2)} | SM: {_truncate(data['obvsm'].values[0], 2)} |"
                f" Diff: {data['obv_sm_diff'].values[0]} | OBVPC: {data['obv_pc'].values[0]}"
                "\n"
                # ADX
                f"ADX14: {_truncate(data['adx14'].values[0], 2)} |"
                f" DI Pcnt: {_truncate(data['di_pcnt'].values[0], 2)}% | +DIpc: {data['+di_pc'].values[0]} |"
                f" +DI14 {_truncate(data['+di14'].values[0], 2)} | -DI14: {_truncate(data['-di14'].values[0], 2)}"
                "\n"
                # MACD
                f"Macd: {_truncate(data['macd'].values[0],6)} |"
                f" Sgnl: {_truncate(data['signal'].values[0],6)} | SigPcnt: {data['macd_sig_pcnt'].values[0]}% |"
                f" PrvSigPcnt: {_df['macd_sig_pcnt'].iloc[-2]}% | Macdpc: {data['macd_pc'].values[0]}"
                "\n"
                # MACD_Leader
                f"MacdLead: {_truncate(data['macdlead'].values[0],6)} | MacdL: {_truncate(data['macdl'].values[0],6)} |"
                f" MacdlSig: {_truncate(data['macdl_sig'].values[0],6)} | MacdLeadpc: {data['macdlead_pc'].values[0]}% | Diff: {data['macdl_sg_diff'].values[0]}%"
                "\n"
#                 # EMA/WMA
#                 f"EMA5pc: {data['ema5_pc'].values[0]} | EMA5: {_truncate(data['ema5'].values[0],2)} | EMA5: {_truncate(data['ema5_wma5'].values[0],2)}"
#                 "\n"    EMAWMA Action: {self.emawma_action} |
                # WilliamsR %
                f"WilliamsR: {data['williamsr14'].values[0]}% | ERI Buy: {data['eri_buy'].values[0]} | ERI Sell: {data['eri_sell'].values[0]}"
                "\n"
                f"SMA5: {_truncate(data['sma5'].values[0],4)} | SMA10: {_truncate(data['sma10'].values[0],4)} | SMA20: {_truncate(data['sma20'].values[0],2)}"
                f" | SMA50: {_truncate(data['sma50'].values[0],4)} | SMA100: {_truncate(data['sma100'].values[0],4)}| SMA200: {_truncate(data['sma200'].values[0],4)}"
                "\n"
                # Actions
                f"Macd Action: {self.macd_action} | MacdL Action: {self.macdl_action} | HybMacd Action: {self.hybmacd_action} | ADX Action: {self.adx_action}"
                "\n"
                f"OBV Action: {self.obv_action} | RSI Action: {self.rsi_action} | WilliamsR Action: {self.williamsr_action} | ERI Action: {self.eri_action}"
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
