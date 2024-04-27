from AlgorithmImports import *
from datetime import datetime, timedelta


# The moving average baseline trading strategy in the cryptocurrency market. 
class CryptoMA(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2019,9,1)    #Set Start Date
        self.SetEndDate(2022,12,31)      #Set End Date
        self.SetCash(1000000)           #Set Strategy Cash
        self.initialCash = 1000000

        ### Instrument
        self.instrument = "BTCBUSD"

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)

        self.symbol = self.AddCrypto(self.instrument, Resolution.Hour, Market.Binance).Symbol
        self.strikethrough = ""
        
        ### Parameters
        self.n_days = 50 #MA
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
        df = self.History(self.symbol, self.n_days, Resolution.Daily)
        # self.Log(f"{'close' in df} {df.shape[0]}")

        # if data is incomplete, exit the function
        if 'close' not in df or df.shape[0] != self.n_days:
            return
        
        # calculate the moving average of the underlying
        MA = df['close'].mean()

        quantity = self.Portfolio[self.symbol].Quantity

        # if the previous price position of the upper line is lower 
        # and the underlying price exceeds the upper MA band (indicating a cross)
        # buy (long) the underlying
        if self.upperlinepos == "Lower" and price >= MA:
            self.SetHoldings(self.symbol,1)
        # if the previous price position of the upper line is upper
        # and the underlying price falls below the lower MA band (indicating a cross)
        # sell (short) the underlying
        if self.upperlinepos == "Upper" and price <= MA:
            self.SetHoldings(self.symbol,-1)
        
        # update the positions of the upper and lower line
        if price >= MA:
            self.upperlinepos = "Upper"
        else:
            self.upperlinepos = "Lower"
        
        if price <= MA:
            self.lowerlinepos = "Lower"
        else:
            self.lowerlinepos = "Upper"
