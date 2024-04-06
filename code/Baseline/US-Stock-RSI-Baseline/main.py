from AlgorithmImports import *
from datetime import datetime, timedelta
import talib as tb


class USStockRSI(QCAlgorithm):

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
        self.buy_rsi = 0.3
        self.sell_rsi = 0.7
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

        if self.symbol in slice.Bars:
            trade_bar = slice.Bars[self.symbol]
            price = trade_bar.Close
            high = trade_bar.High
            low = trade_bar.Low
        else:
            return
        
        history = self.History(self.symbol, self.n_days, Resolution.Daily)
        # self.Log(f"{'close' in df} {df.shape[0]}")
        if 'close' not in history or history.shape[0] != self.n_days:
            return
        history["RSI"] = tb.RSI(history["close"], timeperiod=14)/100
        RSI = history["RSI"].iloc[-1]

        quantity = self.Portfolio[self.symbol].Quantity

        if RSI <= self.buy_rsi:
            self.SetHoldings(self.symbol,1)
        elif RSI >= self.sell_rsi:
            self.SetHoldings(self.symbol,-1)
