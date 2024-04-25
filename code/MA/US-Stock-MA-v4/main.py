from AlgorithmImports import *
from datetime import datetime, timedelta


# Version 4 of the algorithm in the US stock market. 
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
        self.takeprofit = True
        self.takeprofitpercentage = 0.05
        self.trailingstoploss = True
        self.trailingstoplosspercent = 0.03
        # self.percent_above = 0.03
        self.volatility_n_days = 25
        self.volatility_coefficient = 0.3
        self.close_n_days = 100 #MA
        self.close_percent_above = True
        if self.close_percent_above:
            self.close_volatility_n_days = 30
            self.close_volatility_coefficient = 0.4
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

        # obtain the past history of the underlying
        df2 = self.History(self.symbol, self.close_n_days, Resolution.Daily)
        # self.Log(f"{'close' in df} {df.shape[0]}")
        
        # if data is incomplete, exit the function
        if 'close' not in df2 or df2.shape[0] != self.close_n_days:
            return
        
        # calculate the moving average of the underlying to close trades
        close_MA = df2['close'].mean()

        quantity = self.Portfolio[self.symbol].Quantity

        # obtain the past volatility of the underlying stock
        df3 = self.History(self.symbol, self.volatility_n_days, Resolution.Daily)
        
        # determine the width of the MA bands based on past volatility
        self.percent_above = df3['close'].pct_change().dropna().std() * self.volatility_coefficient
        
        # determine the width of the MA bands to close trades based on past volatility
        if self.close_percent_above:
            df4 = self.History(self.symbol, self.close_volatility_n_days, Resolution.Daily)
            self.close_above = df4['close'].pct_change().dropna().std() * self.close_volatility_coefficient

        # check for conditions to close the position and execute market orders
        # the MA bands are used to close the positions in this version
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
                if self.close_percent_above:
                    threshold = close_MA*(1+self.close_above)
                else:
                    threshold = close_MA
                if price <= threshold:
                    self.Log("Pass Close MA: sell")
                    ticket = self.MarketOrder(self.symbol, -quantity)
                    if ticket.QuantityFilled != -quantity:
                        self.cont_liquidate = True
            elif self.strikethrough == "Lower":
                if self.close_percent_above:
                    threshold = close_MA/(1+self.close_above)
                else:
                    threshold = close_MA
                if price >= threshold:
                    self.Log("Pass Close MA: buy back")
                    ticket = self.MarketOrder(self.symbol, -quantity)
                    if ticket.QuantityFilled != -quantity:
                        self.cont_liquidate = True
        # otherwise, trade accordingly
        else:
            self.cont_liquidate = False
            # self.Log(f"{self.upperlinepos} {self.lowerlinepos}")
            
            # if the price position of the upper line is lower
            # and the underlying price exceeds the upper MA band, buy (long) the underlying
            if self.upperlinepos == "Lower" and price >= MA*(1+self.percent_above):
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.symbol, q)
                self.Log(f"Price: {price}, MA: {MA}, buy {q}")
                self.strikethrough = "Upper"
                self.highwatermark = price
                self.lowwatermark = price
                self.cur_purchaseprice = price
            
            # if the price position of the upper line is upper
            # and the underlying price falls below the lower MA band, sell (short) the underlying
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
