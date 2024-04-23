from AlgorithmImports import *
from datetime import datetime, timedelta
import talib as tb


# The MACD baseline trading strategy in the US Stock market. 
class USStockMACD(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2013,1,1)    #Set Start Date
        self.SetEndDate(2022,12,31)      #Set End Date
        self.SetCash(1000000)           #Set Strategy Cash
        self.initialCash = 1000000

        ### Instrument
        self.instrument = "SPY"
        self.SetBenchmark("SPY")

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        self.symbol = self.AddEquity(self.instrument, Resolution.Hour).Symbol
        self.strikethrough = ""
        
        ### Parameters
        self.n_days = 100
        ### End Parameters


        self.upperlinepos = ""
        self.lowerlinepos = ""
        self.cur_purchaseprice = 0
        self.cont_liquidate = False
        self.highwatermark = 0


    def OnData(self, slice:Slice):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''

        # first, check the existence of the data
        if self.symbol in slice.Bars:
            trade_bar = slice.Bars[self.symbol]
            price = trade_bar.Close
            high = trade_bar.High
            low = trade_bar.Low
        # if data does not exist, exit this function
        else:
            return
        
        # obtain the past history of the underlying
        history = self.History(self.symbol, self.n_days, Resolution.Daily)
        # self.Log(f"{'close' in df} {df.shape[0]}")

        # if data is incomplete, exit the function
        if 'close' not in history or history.shape[0] != self.n_days:
            return

        # obtain the MA, MACD and the corresponding signals
        history["MA50"] = tb.MA(history["close"], timeperiod=50)
        history["MACD"] = tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
        history["MACD_Signal"] = tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[1]
        last_signal = history["MACD_Signal"].iloc[-2]
        current_signal = history["MACD_Signal"].iloc[-1]
        last_macd = history["MACD"].iloc[-2]
        current_macd = history["MACD"].iloc[-1]

        # if there is a MACD crossover from below the signal line, go long
        if last_macd < last_signal and current_signal < current_macd:
            self.SetHoldings(self.symbol,1)
        # otherwise if there is a MACD crossover from above the signal line, go short
        elif last_macd > last_signal and current_signal > current_macd:
            self.SetHoldings(self.symbol,-1)
