# region imports
from AlgorithmImports import *
# endregion
import talib as tb
from sklearn import svm
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# The machine learning baseline trading strategy with the use of the support vector machine (SVM) classifier in the cryptocurrency market.
# NOTE: This algorithm is not published in the report and final results, 
# and it is only involved in the preliminary development process. 
class ML(QCAlgorithm):

    def TrainAlgo(self):
        '''Trains the algorithm with the machine learning/deep learning model. '''
        # obtains the past history
        history = self.History(self.stock, self.tradeHistory, self.setResolution)

        # if the history is incomplete, early exit the function
        if history.shape[0] != self.tradeHistory:
            return
        
        # obtain different types of technical indicators as the input to the model
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
        
        # define a scaler to scale all the raw inputs
        self.scaler = StandardScaler()

        # training
        # create an instance of the model and build its architecture
        self.model = svm.SVC(kernel=self.kernel)
        X = history[["LMA50","LMA100","LMA200","RSI","MACD","pc"]]
        X = self.scaler.fit_transform(X)
        y = history["signal"]
        self.model.fit(X,y)
        self.model_trained = True
        return

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. 
        All algorithms must be initialized before performing testing.'''
        self.SetStartDate(2019, 9, 1)  # Set Start Date
        self.SetEndDate(2022,12,31)
        self.SetCash(1000000)  # Set Strategy Cash
        
        #params
        self.setResolution = Resolution.Daily
        self.tradePeriod = 25
        self.traded = False
        self.placedTrade = self.Time
        self.tradeConfidenceLevel = 0.6
        self.tradeHistory = 365
        self.npercent = 0.1
        self.stoplosspercent = 0.04 #0 for no stoploss
        self.sellatnpercent = False
        self.sellatndays = True
        self.extendWhenSignalled = False 
        self.updateExePriceWhenExtend = True # self.extendWhenSignalled must be true
        self.sellWhenSignalChange = True 

        #ML specific param
        self.kernel = "rbf"

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.stock = self.AddCrypto("BTCBUSD", self.setResolution, Market.Binance).Symbol

        #train
        self.model_trained = False
        self.TrainAlgo()

    def OnData(self, data: Slice):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''

        # first, check the existence of the data
        # if data does not exist, early exit this function
        if self.stock in data.Bars:
            trade_bar = data.Bars[self.stock]
            price = trade_bar.Close
            high = trade_bar.High
            low = trade_bar.Low
        else:
            return
        
        # train the model if the model is not (properly) trained
        if not self.model_trained:
            self.TrainAlgo()
            return

        # obtain the past history of data and obtain technical indicators
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
        
        # concatenate all the technical indicators into a DataFrame (or a 2D numpy array)
        X = [history.iloc[-1][["LMA50","LMA100","LMA200","RSI","MACD","pc"]]]
        X = self.scaler.transform(X)

        # obtain predictions from the model and generate signals
        signal = self.model.predict(X)

        q = self.Portfolio[self.stock].Quantity

        # trade based on the model predictions and a set of defined algorithm parameters
        if abs(q*price)>100:
            # update the execution price if not initialized
            if self.exePrice == 0:
                self.exePrice = self.Portfolio[self.stock].AveragePrice
            sell = False
            
            # obtain the date of the last trade, and to update signals accordingly
            his = self.History(self.stock, self.tradePeriod+1, self.setResolution)
            # self.Log(his)
            # self.Log(his.index[0])
            
            # if the number of days since the last trade expires, it will exit the trade positions
            if self.sellatndays:
                # delta = timedelta(hours=self.tradePeriod) if self.setResolution == Resolution.Hour else timedelta(days=self.tradePeriod)
                if self.placedTrade <= his.index[0][1].strftime("%Y-%m-%d"):
                    sell = True
                    self.Log(f"Time expired: {his.index[0][1].strftime('%Y-%m-%d')}")
            
            # if there is update in the signal, extend the holding period
            if self.extendWhenSignalled:
                if (signal >= self.tradeConfidenceLevel and q > 0) or (signal <= -self.tradeConfidenceLevel and q < 0):
                    self.placedTrade = self.Time.strftime("%Y-%m-%d") #extend
                    if self.updateExePriceWhenExtend:
                        self.exePrice = price
                    self.Log(f"Time extended")
            
            # if there are changes in the trading signal, update the sell signal
            if self.sellWhenSignalChange:
                if (signal >= self.tradeConfidenceLevel and q < 0) or (signal <= -self.tradeConfidenceLevel and q > 0):
                    sell = True
                    self.Log(f"Signal Changed: sell")

            # if it is not a sell signal and set to be sold at a fixed percentage (n%), trade
            if not sell and self.sellatnpercent:
                self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
                # if the portfolio quantity is positive, execute a sell order
                if q>0:
                    if price > self.exePrice * (1 + self.npercent):
                        self.MarketOrder(self.stock, -q)
                    else:
                        self.LimitOrder(self.stock, -q, self.exePrice * (1 + self.npercent) , tag="take profit")
                # otherwise if the portfolio quantity is negative, execute a buy order
                else:
                    if price < self.exePrice * (1 - self.npercent):
                        self.MarketOrder(self.stock, -q)
                    else:
                        self.LimitOrder(self.stock, -q, self.exePrice * (1 - self.npercent) , tag="take profit")

            # if it is not a sell signal and set to have a stop loss percentage, trade
            if not sell and self.stoplosspercent>0:
                self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
                # if the portfolio quantity is positive, execute a stop loss sell order
                if q>0:
                    if price < self.exePrice * (1 - self.stoplosspercent):
                        self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                        self.MarketOrder(self.stock, -q)
                        self.Log(f"Stop Loss, price: {price}")
                    else:
                        self.StopLimitOrder(self.stock, -q, self.exePrice * (1 - self.stoplosspercent), self.exePrice * (1 - self.stoplosspercent))
                # otherwise, if the portfolio quantity is negative, execute a stop buy order
                else:
                    if price > self.exePrice * (1 + self.stoplosspercent):
                        self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                        self.MarketOrder(self.stock, -q)
                        self.Log(f"Stop Loss, price: {price}")
                    else:
                        self.StopLimitOrder(self.stock, -q, self.exePrice * (1 + self.stoplosspercent), self.exePrice * (1 + self.stoplosspercent))
            
            # if a sell signal is triggered, execute sell orders
            if sell:
                self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                ticket = self.MarketOrder(self.stock, -q)
                
        # if there are no positions, trade accordingly
        else:
            self.exePrice = 0
            self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
            # place a buy order if a buy signal is triggered
            if signal >= self.tradeConfidenceLevel:
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.stock, q)
                self.Log(f"Price: {price}, signal: {signal}, buy {q}")
                self.traded = True
                self.placedTrade = self.Time.strftime("%Y-%m-%d")
            # place a sell order is a sell signal is triggered
            elif signal <= -self.tradeConfidenceLevel:
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.stock, -q)
                self.Log(f"Price: {price}, signal: {signal}, sell {q}")
                self.traded = True
                self.placedTrade = self.Time.strftime("%Y-%m-%d")
