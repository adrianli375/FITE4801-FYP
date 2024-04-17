from AlgorithmImports import *
from datetime import datetime, timedelta


class StockMA(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2013,1,1)    #Set Start Date
        self.SetEndDate(2022,12,31)      #Set End Date
        self.SetCash(1000000)           #Set Strategy Cash
        self.initialCash = 1000000

        ### Instrument
        self.instrument = "SPY"

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        self.symbol = self.AddEquity(self.instrument, Resolution.Hour).Symbol
        self.strikethrough = ""
        
        ### Parameters
        self.n_days = 100 #MA
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
        
        df = self.History(self.symbol, self.n_days, Resolution.Daily)
        # self.Log(f"{'close' in df} {df.shape[0]}")
        if 'close' not in df or df.shape[0] != self.n_days:
            return
        MA = df['close'].mean()

        quantity = self.Portfolio[self.symbol].Quantity

        # if abs(quantity)*price > 10:
        #     if self.cont_liquidate:
        #         ticket = self.MarketOrder(self.symbol, -quantity)
        #         if ticket.QuantityFilled != -quantity:
        #             self.cont_liquidate = True
        #     elif self.strikethrough == "Upper":
        #         if price <= MA:
        #             self.Log("Pass MA: sell")
        #             ticket = self.MarketOrder(self.symbol, -quantity)
        #             if ticket.QuantityFilled != -quantity:
        #                 self.cont_liquidate = True
        #     elif self.strikethrough == "Lower":
        #         if price >= MA:
        #             self.Log("Pass MA: buy back")
        #             ticket = self.MarketOrder(self.symbol, -quantity)
        #             if ticket.QuantityFilled != -quantity:
        #                 self.cont_liquidate = True
        # else:
        #     self.cont_liquidate = False
        #     # self.Log(f"{self.upperlinepos} {self.lowerlinepos}")
        if self.upperlinepos == "Lower" and price >= MA:
            self.SetHoldings(self.symbol,1)
        if self.upperlinepos == "Upper" and price <= MA:
            self.SetHoldings(self.symbol,-1)
        
        if price >= MA:
            self.upperlinepos = "Upper"
        else:
            self.upperlinepos = "Lower"
        
        # if price <= MA:
        #     self.lowerlinepos = "Lower"
        # else:
        #     self.lowerlinepos = "Upper"
