# region imports
from AlgorithmImports import *
# endregion
import talib as tb
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class ML(QCAlgorithm):
    def TrainAlgo(self):
        history = self.History(self.stock, self.tradeHistory, self.setResolution)
        if  history.shape[0] != self.tradeHistory:
            return
        history["MA50"] = tb.MA(history["close"], timeperiod=50)
        history["MA100"] = tb.MA(history["close"], timeperiod=100)
        history["MA200"] = tb.MA(history["close"], timeperiod=200)
        history["MACD"] =  tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
        history["RSI"] =  tb.RSI(history["close"], timeperiod=14)/100
        history["lc"] = history["close"].shift(1)
        history["pc"] = history["close"]/history["lc"] - 1 
        history["n5c"] = history["open"].shift(-self.tradePeriod)
        history["n5pc"] = history["n5c"]/history["open"] - 1
        history.dropna(inplace=True)
        history["LMA50"] = (history["close"] - history["MA50"])/history["close"]
        history["LMA100"] = (history["close"] - history["MA100"])/history["close"]
        history["LMA200"] = (history["close"] - history["MA200"])/history["close"]
        history["signal"] =  history["n5pc"].apply(lambda x: 1 if x > self.npercent else ( -1 if x < -self.npercent else 0))   
        
        self.scaler = StandardScaler()

        #training
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        X = history[["LMA50","LMA100","LMA200","RSI","MACD","pc"]]
        X = self.scaler.fit_transform(X)
        y = history["signal"]
        self.model.fit(X,y)
        self.model_trained = True
        return

    def Initialize(self):
        self.SetStartDate(2020, 9, 1)  # Set Start Date
        self.SetEndDate(2022,12,31)
        self.SetCash(1000000)  # Set Strategy Cash
        
        #params
        self.setResolution = Resolution.Daily
        self.tradePeriod = 20
        self.traded = False
        self.placedTrade = self.Time
        self.tradeConfidenceLevel = 0.6
        self.tradeHistory = 365
        self.npercent = 0.05
        self.stoplosspercent = 0.05 #0 for no stoploss
        self.sellatnpercent = True
        self.sellatndays = False
        self.extendWhenSignalled = False 
        self.updateExePriceWhenExtend = False # self.extendWhenSignalled must be true
        self.sellWhenSignalChange = True 

        #ML specific param
        self.n_neighbors = 5

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.stock = self.AddCrypto("BTCBUSD", self.setResolution, Market.Binance).Symbol

        #train
        self.model_trained = False
        self.TrainAlgo()

    def OnData(self, data: Slice):
        if self.stock in data.Bars:
            trade_bar = data.Bars[self.stock]
            price = trade_bar.Close
            high = trade_bar.High
            low = trade_bar.Low
        else:
            return
        
        if not self.model_trained:
            self.TrainAlgo()
            return

        history = self.History(self.stock, 205, self.setResolution)
        history["MA50"] = tb.MA(history["close"], timeperiod=50)
        history["MA100"] = tb.MA(history["close"], timeperiod=100)
        history["MA200"] = tb.MA(history["close"], timeperiod=200)
        history["MACD"] =  tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
        history["RSI"] =  tb.RSI(history["close"], timeperiod=14)/100
        history["lc"] = history["close"].shift(1)
        history["pc"] = history["close"]/history["lc"] - 1 
        # history["n5c"] = history["close"].shift(-self.tradePeriod)
        # history["n5pc"] = history["n5c"]/history["close"] - 1
        history.dropna(inplace=True)
        history["LMA50"] = (history["close"] - history["MA50"])/history["close"]
        history["LMA100"] = (history["close"] - history["MA100"])/history["close"]
        history["LMA200"] = (history["close"] - history["MA200"])/history["close"]
        # history["signal"] =  history["n5pc"].apply(lambda x: 1 if x > 0.03 else ( -1 if x < -0.03 else 0))
        X = [history.iloc[-1][["LMA50","LMA100","LMA200","RSI","MACD","pc"]]]
        X = self.scaler.transform(X)
        signal = self.model.predict(X)

        q = self.Portfolio[self.stock].Quantity

        if abs(q*price)>100:
            if self.exePrice == 0:
                self.exePrice = self.Portfolio[self.stock].AveragePrice
            sell = False
            his = self.History(self.stock, self.tradePeriod+1, self.setResolution)
            # self.Log(his)
            # self.Log(his.index[0])
            if self.sellatndays:
                # delta = timedelta(hours=self.tradePeriod) if self.setResolution == Resolution.Hour else timedelta(days=self.tradePeriod)
                if self.placedTrade <= his.index[0][1].strftime("%Y-%m-%d"):
                    sell = True
                    self.Log(f"Time expired: {his.index[0][1].strftime('%Y-%m-%d')}")
            
            if self.extendWhenSignalled:
                if (signal >= self.tradeConfidenceLevel and q > 0) or (signal <= -self.tradeConfidenceLevel and q < 0):
                    self.placedTrade = self.Time.strftime("%Y-%m-%d") #extend
                    if self.updateExePriceWhenExtend:
                        self.exePrice = price
                    self.Log(f"Time extended")
            
            if self.sellWhenSignalChange:
                if (signal >= self.tradeConfidenceLevel and q < 0) or (signal <= -self.tradeConfidenceLevel and q > 0):
                    sell = True
                    self.Log(f"Signal Changed: sell")

            if not sell and self.sellatnpercent:
                self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
                if q>0:
                    self.LimitOrder(self.stock, -q, self.exePrice * (1 + self.npercent) , tag="take profit")
                else:
                    self.LimitOrder(self.stock, -q, self.exePrice * (1 - self.npercent) , tag="take profit")

            if not sell and self.stoplosspercent>0:
                self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
                if q>0:
                    if price < self.exePrice * (1 - self.stoplosspercent):
                        self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                        self.MarketOrder(self.stock, -q)
                        self.Log(f"Stop Loss, price: {price}")
                    else:
                        self.StopLimitOrder(self.stock, -q, self.exePrice * (1 - self.stoplosspercent), self.exePrice * (1 - self.stoplosspercent))
                else:
                    if price > self.exePrice * (1 + self.stoplosspercent):
                        self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                        self.MarketOrder(self.stock, -q)
                        self.Log(f"Stop Loss, price: {price}")
                    else:
                        self.StopLimitOrder(self.stock, -q, self.exePrice * (1 + self.stoplosspercent), self.exePrice * (1 + self.stoplosspercent))
            
            if sell:
                self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                ticket = self.MarketOrder(self.stock, -q)
                
        else:
            self.exePrice = 0
            self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
            if signal >= self.tradeConfidenceLevel:
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.stock, q)
                self.Log(f"Price: {price}, signal: {signal}, buy {q}")
                self.traded = True
                self.placedTrade = self.Time.strftime("%Y-%m-%d")
            elif signal <= -self.tradeConfidenceLevel:
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.stock, -q)
                self.Log(f"Price: {price}, signal: {signal}, sell {q}")
                self.traded = True
                self.placedTrade = self.Time.strftime("%Y-%m-%d")
