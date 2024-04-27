from AlgorithmImports import *
from datetime import datetime, timedelta


# Version 1 of the algorithm in the cryptocurrency market. 
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
        self.takeprofit = False
        self.takeprofitpercentage = 0.4
        self.trailingstoploss = False
        self.trailingstoplosspercent = 0.07
        self.percent_above = 0.03
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

        # check for conditions to close the position and execute market orders
        if abs(quantity)*price > 10:
            self.highwatermark = max(price,self.highwatermark)
            self.lowwatermark = min(price,self.lowwatermark)
            if self.cont_liquidate:
                ticket = self.MarketOrder(self.symbol, -quantity)
                if ticket.QuantityFilled != -quantity:
                    self.cont_liquidate = True
            elif self.takeprofit and ((self.strikethrough == "Upper" and price/self.cur_purchaseprice - 1 > self.takeprofitpercentage) or (self.strikethrough == "Lower" and 1 - price/self.cur_purchaseprice > self.takeprofitpercentage)):
                self.Log("Take Profit {},{},{},{}".format(price,MA,self.strikethrough,quantity))
                ticket = self.MarketOrder(self.symbol, -quantity)
                if ticket.QuantityFilled != -quantity:
                    self.cont_liquidate = True
            elif self.trailingstoploss and ((self.strikethrough == "Upper" and price/self.highwatermark < 1 - self.trailingstoplosspercent) or (self.strikethrough == "Lower" and price/self.lowwatermark > 1 + self.trailingstoplosspercent)):
                self.Log("Trailing stop loss: price: {}, highwm:{}, lowwm:{},{}, q:{}".format(price,self.highwatermark,self.lowwatermark,self.strikethrough,quantity))
                ticket = self.MarketOrder(self.symbol, -quantity)
                if ticket.QuantityFilled != -quantity:
                    self.cont_liquidate = True
            elif self.strikethrough == "Upper":
                if price <= MA:
                    self.Log("Pass MA: sell")
                    ticket = self.MarketOrder(self.symbol, -quantity)
                    if ticket.QuantityFilled != -quantity:
                        self.cont_liquidate = True
            elif self.strikethrough == "Lower":
                if price >= MA:
                    self.Log("Pass MA: buy back")
                    ticket = self.MarketOrder(self.symbol, -quantity)
                    if ticket.QuantityFilled != -quantity:
                        self.cont_liquidate = True
        # otherwise, trade accordingly
        else:
            self.cont_liquidate = False
            # self.Log(f"{self.upperlinepos} {self.lowerlinepos}")

            # if the previous price position of the upper line is lower 
            # and the underlying price exceeds the upper MA band (indicating a cross)
            # buy (long) the underlying
            if self.upperlinepos == "Lower" and price >= MA*(1+self.percent_above):
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.symbol, q)
                self.Log(f"Price: {price}, MA: {MA}, buy {q}")
                self.strikethrough = "Upper"
                self.highwatermark = price
                self.lowwatermark = price
                self.cur_purchaseprice = price
            
            # if the previous price position of the upper line is upper
            # and the underlying price falls below the lower MA band (indicating a cross)
            # sell (short) the underlying
            if self.lowerlinepos == "Upper" and price <= MA/(1+self.percent_above):
                q = self.Portfolio.Cash/price
                ticket = self.MarketOrder(self.symbol, -q)
                self.Log(f"Price: {price}, MA: {MA}, sell {q}")
                self.strikethrough = "Lower"
                self.highwatermark = price
                self.lowwatermark = price
                self.cur_purchaseprice = price
        
        # update the positions of the upper and lower line
        if price >= MA*(1+self.percent_above):
            self.upperlinepos = "Upper"
        else:
            self.upperlinepos = "Lower"
        
        if price <= MA/(1+self.percent_above):
            self.lowerlinepos = "Lower"
        else:
            self.lowerlinepos = "Upper"
