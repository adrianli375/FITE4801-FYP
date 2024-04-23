from AlgorithmImports import *
from datetime import datetime, timedelta
import talib as tb


# The RSI baseline trading strategy in the cryptocurrency market. 
class CryptoRSI(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2019,9,1)    #Set Start Date
        self.SetEndDate(2022,12,31)      #Set End Date
        self.SetCash(1000000)           #Set Strategy Cash
        self.initialCash = 1000000

        ### Instrument
        self.instrument = "BTCUSDT"

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)

        self.symbol = self.AddCrypto(self.instrument, Resolution.Hour, Market.Binance).Symbol
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
        
        # obtain the RSI value
        history["RSI"] = tb.RSI(history["close"], timeperiod=14)/100
        RSI = history["RSI"].iloc[-1]

        # if the RSI value is below a threshold, go long
        if RSI <= self.buy_rsi:
            self.SetHoldings(self.symbol,1)
        # otherwise, if the RSI value is above a threshold, go short
        elif RSI >= self.sell_rsi:
            self.SetHoldings(self.symbol,-1)
