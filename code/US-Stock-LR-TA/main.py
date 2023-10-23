# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import talib as tb
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta


class MLStrategyLR(QCAlgorithm):

    def TrainAlgo(self):
        history = self.History(self.stock, self.tradeHistory, self.setResolution)
        history["MA50"] = tb.MA(history["close"], timeperiod=50)
        history["MA100"] = tb.MA(history["close"], timeperiod=100)
        history["MA200"] = tb.MA(history["close"], timeperiod=200)
        history["Cross"] = self.create_cross_feature(history["MA50"], history["MA200"])

        history["MACD"] =  tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
        history["RSI"] =  tb.RSI(history["close"], timeperiod=14)
        history["RSIOverbought"] = history["RSI"].apply(lambda x: 1 if x > 70 else 0)
        history["RSIOversold"] = history["RSI"].apply(lambda x: 1 if x < 30 else 0)

        history["lc"] = history["close"].shift(1)
        history["pc"] = history["close"]/history["lc"] - 1 
        history["n5c"] = history["close"].shift(-self.tradePeriod)
        history["n5pc"] = history["n5c"]/history["close"] - 1
        history.dropna(inplace=True)
        history["LMA50"] = (history["close"] - history["MA50"])/history["close"]
        history["LMA100"] = (history["close"] - history["MA100"])/history["close"]
        history["LMA200"] = (history["close"] - history["MA200"])/history["close"]
        history["signal"] =  history["n5pc"].apply(lambda x: 1 if x > 0.03 else ( -1 if x < -0.03 else 0))   
        
        #training
        self.model = LogisticRegression()
        X = history[["LMA50","LMA100","LMA200","Cross","RSIOverbought","RSIOversold","MACD","pc"]]
        y = history["signal"]
        self.model.fit(X,y)
        return

    def Initialize(self):
        self.SetStartDate(2013, 1, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31)
        self.SetCash(1000000)  # Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        #params
        self.setResolution = Resolution.Daily
        self.tradePeriod = 5
        self.traded = False
        self.placedTrade = self.Time
        self.marginforsafety = 0.8
        self.tradeConfidenceLevel = 0.6
        self.tradeHistory = 250

        self.stock = self.AddEquity("SPY", self.setResolution).Symbol

        #train
        self.TrainAlgo()



    def OnData(self, data: Slice):
        if self.stock in data.Bars:
            trade_bar = data.Bars[self.stock]
            price = trade_bar.Close
            high = trade_bar.High
            low = trade_bar.Low
        else:
            return

        if self.traded:
            delta = timedelta(hours=self.tradePeriod) if self.setResolution == Resolution.Hour else timedelta(days=self.tradePeriod)
            if self.Time >= self.placedTrade + delta:
                self.Liquidate()
                self.traded = False
        else:
            history = self.History(self.stock, 205, self.setResolution)
            history["MA50"] = tb.MA(history["close"], timeperiod=50)
            history["MA100"] = tb.MA(history["close"], timeperiod=100)
            history["MA200"] = tb.MA(history["close"], timeperiod=200)
            history["Cross"] = self.create_cross_feature(history["MA50"], history["MA200"])

            history["MACD"] =  tb.MACD(history["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
            history["RSI"] =  tb.RSI(history["close"], timeperiod=14)
            history["RSIOverbought"] = history["RSI"].apply(lambda x: 1 if x > 70 else 0)
            history["RSIOversold"] = history["RSI"].apply(lambda x: 1 if x < 30 else 0)

            history["lc"] = history["close"].shift(1)
            history["pc"] = history["close"]/history["lc"] - 1 
            history["n5c"] = history["close"].shift(-self.tradePeriod)
            history["n5pc"] = history["n5c"]/history["close"] - 1
            history.dropna(inplace=True)
            history["LMA50"] = (history["close"] - history["MA50"])/history["close"]
            history["LMA100"] = (history["close"] - history["MA100"])/history["close"]
            history["LMA200"] = (history["close"] - history["MA200"])/history["close"]
            X = [history.iloc[-1][["LMA50","LMA100","LMA200","Cross","RSIOverbought","RSIOversold","MACD","pc"]]]
            self.Log(X)
            signal = self.model.predict(X)
            self.Log(signal)
            if signal >= self.tradeConfidenceLevel:
                q = self.Portfolio.MarginRemaining/price * self.marginforsafety
                ticket = self.MarketOrder(self.stock, q)
                self.Log(f"Price: {price}, signal: {signal}, buy {q}")
                self.traded = True
                self.placedTrade = self.Time
            elif signal <= -self.tradeConfidenceLevel:
                q = self.Portfolio.MarginRemaining/price * self.marginforsafety
                ticket = self.MarketOrder(self.stock, -q)
                self.Log(f"Price: {price}, signal: {signal}, sell {q}")
                self.traded = True
                self.placedTrade = self.Time
    
    @staticmethod
    def create_cross_feature(ma50, ma200):
        cross_feature = np.zeros_like(ma50)  # Initialize the new feature array with zeros
        
        for i in range(1, len(ma50)):
            if ma50[i] > ma200[i] and ma50[i-1] < ma200[i-1]:
                cross_feature[i] = 1  # Golden Cross
            elif ma50[i] < ma200[i] and ma50[i-1] > ma200[i-1]:
                cross_feature[i] = -1  # Death Cross
        
        return cross_feature
