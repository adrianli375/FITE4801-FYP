from AlgorithmImports import *
from datetime import datetime, timedelta
import talib as tb


class CryptoMACD(QCAlgorithm):

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
        history["MA50"] = tb.MA(history["close"], timeperiod=50)
        history["MACD"] = tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
        history["MACD_Signal"] = tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[1]
        last_signal = history["MACD_Signal"].iloc[-2]
        current_signal = history["MACD_Signal"].iloc[-1]
        last_macd = history["MACD"].iloc[-2]
        current_macd = history["MACD"].iloc[-1]

        quantity = self.Portfolio[self.symbol].Quantity

        if last_macd < last_signal and current_signal < current_macd:
            self.SetHoldings(self.symbol,1)
        elif last_macd > last_signal and current_signal > current_macd:
            self.SetHoldings(self.symbol,-1)
