from AlgorithmImports import *
from datetime import datetime, timedelta


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
        self.n_days = 100 #Number of days used to calculate MA (= max MA)
        self.default_MA = 50
        self.rolling_window = 5
        self.rolling_reset = 10
        self.MA_coef = 2.5
        self.min_MA = 10
        self.takeprofit = False
        self.takeprofitpercentage = 0.4
        self.trailingstoploss = False
        self.trailingstoplosspercent = 0.07
        # self.percent_above = 0.03
        self.volatility_n_days = 5
        self.volatility_coefficient = 0.5

        self.close_volatility_n_days = 20
        self.close_volatility_coefficient = 0.05
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

        ### (v5) Analysis to calculate MA

        lastmax = 0
        lastmaxindex = 0
        reset = True
        high_points = []
        low_points = []
        for i in range(len(df["close"].values)):
            if i > 4:
                curmax = df.iloc[i-self.rolling_window:i+1]["close"].max()
                # print(i,curmax,lastmax,lastmaxindex)
                if curmax > lastmax:
                    if lastmaxindex >= i - self.rolling_window or lastmaxindex == 0:
                        lastmaxindex = i
                        lastmax = curmax
                    else:
                        if not reset:
                            high_points.append(lastmaxindex)
                        lastmaxindex = i
                        lastmax = curmax
                        reset = False
                if i - lastmaxindex > self.rolling_reset or i == len(df["close"].values)-1:
                    if not reset or curmax < lastmax:
                        high_points.append(lastmaxindex)
                    lastmaxindex = 0
                    lastmax = 0
                    reset = True

        lastmax = 1000000000
        lastmaxindex = 0
        reset = True
        for i in range(len(df["close"].values)):
            if i > 4:
                curmax = df.iloc[i-self.rolling_window:i+1]["close"].min()
                # print(i,curmax,lastmax,lastmaxindex)
                if curmax < lastmax:
                    if lastmaxindex >= i - self.rolling_window or lastmaxindex == 1000000000:
                        lastmaxindex = i
                        lastmax = curmax
                    else:
                        if not reset:
                            low_points.append(lastmaxindex)
                        lastmaxindex = i
                        lastmax = curmax
                        reset = False
                if i - lastmaxindex > self.rolling_reset or i == len(df["close"].values)-1:
                    if not reset or curmax > lastmax:
                        low_points.append(lastmaxindex)
                    lastmaxindex = 0
                    lastmax = 1000000000
                    reset = True

        high_diff = []
        if len(high_points) >= 2:
            for i in range(1,len(high_points)):
                high_diff.append(high_points[i]-high_points[i-1])

        low_diff = []
        if len(low_points) >= 2:
            for i in range(1,len(low_points)):
                low_diff.append(low_points[i]-low_points[i-1])
        
        if len(high_diff) == 0:
            high_MA = self.default_MA
        else:
            high_MA = sum(high_diff) / len(high_diff)

        if len(low_diff) == 0:
            low_MA = self.default_MA
        else:
            low_MA = sum(low_diff) / len(low_diff)
        
        final_MA = int((high_MA + low_MA)/2*self.MA_coef)

        if final_MA < self.min_MA:
            final_MA = self.min_MA

        df = self.History(self.symbol, final_MA, Resolution.Daily)
        MA = df['close'].mean()

        close_MA = MA

        quantity = self.Portfolio[self.symbol].Quantity

        df3 = self.History(self.symbol, self.volatility_n_days, Resolution.Daily)
        self.percent_above = df3['close'].pct_change().dropna().std() * self.volatility_coefficient
        
        df4 = self.History(self.symbol, self.close_volatility_n_days, Resolution.Daily)
        self.close_above = df4['close'].pct_change().dropna().std() * self.close_volatility_coefficient

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
                threshold = close_MA*(1+self.close_above)
                if price <= threshold:
                    self.Log("Pass Close MA: sell")
                    ticket = self.MarketOrder(self.symbol, -quantity)
                    if ticket.QuantityFilled != -quantity:
                        self.cont_liquidate = True
            elif self.strikethrough == "Lower":
                
                threshold = close_MA/(1+self.close_above)
                if price >= threshold:
                    self.Log("Pass Close MA: buy back")
                    ticket = self.MarketOrder(self.symbol, -quantity)
                    if ticket.QuantityFilled != -quantity:
                        self.cont_liquidate = True
        else:
            self.cont_liquidate = False
            # self.Log(f"{self.upperlinepos} {self.lowerlinepos}")
            if self.upperlinepos == "Lower" and price >= MA*(1+self.percent_above):
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.symbol, q)
                self.Log(f"Price: {price}, MA: {MA}, buy {q}")
                self.strikethrough = "Upper"
                self.highwatermark = price
                self.lowwatermark = price
                self.cur_purchaseprice = price
            if self.lowerlinepos == "Upper" and price <= MA/(1+self.percent_above):
                q = self.Portfolio.Cash/price
                ticket = self.MarketOrder(self.symbol, -q)
                self.Log(f"Price: {price}, MA: {MA}, sell {q}")
                self.strikethrough = "Lower"
                self.highwatermark = price
                self.lowwatermark = price
                self.cur_purchaseprice = price
        
        if price >= MA*(1+self.percent_above):
            self.upperlinepos = "Upper"
        else:
            self.upperlinepos = "Lower"
        
        if price <= MA/(1+self.percent_above):
            self.lowerlinepos = "Lower"
        else:
            self.lowerlinepos = "Upper"
