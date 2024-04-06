# region imports
from AlgorithmImports import *
# endregion
import talib as tb
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense, TimeDistributed

BUY_SIGNAL = 1
HOLD_SIGNAL = 0
SELL_SIGNAL = 2

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
        # requirement set by the model: the class name must be non negative integers
        history["signal"] =  history["pc"].apply(lambda x: BUY_SIGNAL if x > self.npercent else (SELL_SIGNAL if x < -self.npercent else HOLD_SIGNAL))   
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.output_scaler = MinMaxScaler(feature_range=(-1, 1))

        #training
        X = history[["LMA50","LMA100","LMA200","RSI","MACD"]]
        X = self.preprocess_X(X, train=True)
        if X is None:
            return
        
        # signal: classification, close: regression
        y_raw = history["lc"][self.modelpastndays:].values.reshape(-1, 1)
        y = np.squeeze(self.output_scaler.fit_transform(y_raw))

        self.model = Sequential()
        self.model.add(Dense(256, activation='tanh', input_shape=(X.shape[1], X.shape[2])))
        self.model.add(Dropout(0.05))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dropout(0.1))
        self.model.add(GRU(self.gru_units, activation='tanh', return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dropout(0.1))
        self.model.add(GRU(self.gru_units, activation='tanh'))
        self.model.add(Dropout(0.1))
        # tanh: regression, softmax: classification
        self.model.add(Dense(1))
        # classification: metrics=['accuracy'], loss=sparse_categorical_crossentropy
        # regression: metrics=['mean_squared_error'], loss=mean_squared_error
        self.model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error'])

        self.model.fit(X,y)
        self.model_trained = True
        self.Log(f'Model trained at {self.Time}')
        return

    def preprocess_X(self, X, train=False):
        output = []
        if X.shape[0] == 0:
            return None
        if train:
            scaled = self.scaler.fit_transform(X)
        else:
            scaled = self.scaler.transform(X)
        n = scaled.shape[0]
        d = X.shape[1]
        for j in range(d):
            output.append([])
            for i in range(self.modelpastndays, n):
                output[j].append(scaled[i-self.modelpastndays:i, j])
        if not any(output):
            return None
        try:
            output = np.moveaxis(output, [0], [2])
            self.Log(output.shape)
            return output
        except Exception as e:
            return None
        
    def Initialize(self):
        self.SetStartDate(2019, 9, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31)
        self.SetCash(1000000)  # Set Strategy Cash
        
        #params
        self.setResolution = Resolution.Hour
        self.tradePeriod = 12
        self.traded = False
        self.placedTrade = self.Time
        self.tradeConfidenceLevel = 0.4
        self.tradeHistory = 60 * 24
        self.npercent = 0.05
        self.stoplosspercent = 0.1 #0 for no stoploss
        self.sellatnpercent = True
        self.sellatndays = True
        self.extendWhenSignalled = False
        self.updateExePriceWhenExtend = False # self.extendWhenSignalled must be true
        self.sellWhenSignalChange = True
        self.modelTrainFrequency = 30 # this is in days

        #ML specific param
        self.modelpastndays = 5 * 24
        self.gru_units = 64

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.stock = self.AddCrypto("BTCUSDT", self.setResolution).Symbol

        #train
        self.model_trained = False
        self.days_since_last_train = 0
        self.TrainAlgo()

    def OnData(self, data: Slice):
        if self.stock in data.Bars:
            trade_bar = data.Bars[self.stock]
            price = trade_bar.Close
            high = trade_bar.High
            low = trade_bar.Low
        else:
            return
        
        if not self.model_trained or self.days_since_last_train >= self.modelTrainFrequency:
            self.TrainAlgo()
            self.days_since_last_train = 0
            return
        self.days_since_last_train += 1

        history = self.History(self.stock, 200 + self.modelpastndays, self.setResolution)
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
        X = history[["LMA50","LMA100","LMA200","RSI","MACD"]]
        X = self.preprocess_X(X, train=False)
        if X is None:
            return
        X_test = np.array([X[-1]])

        raw_pred = self.model.predict(X_test)
        y_pred = self.output_scaler.inverse_transform(raw_pred)[0][0]
        raw_pred_value = raw_pred[0][0]
        signal = BUY_SIGNAL if y_pred > price * (1 + self.npercent) else (SELL_SIGNAL if y_pred < price * (1 - self.npercent) else HOLD_SIGNAL)

        q = self.Portfolio[self.stock].Quantity

        if abs(q*price)>100 and self.Portfolio.MarginRemaining > q * price:
            if self.exePrice == 0:
                self.exePrice = self.Portfolio[self.stock].AveragePrice
            sell = False
            his = self.History(self.stock, self.tradePeriod+1, self.setResolution)
            # self.Log(his)
            # self.Log(his.index[0])
            if self.sellatndays:
                # delta = timedelta(hours=self.tradePeriod) if self.setResolution == Resolution.Hour else timedelta(days=self.tradePeriod)
                if self.placedTrade <= his.index[0][1].strftime("%Y-%m-%d %H:%M"):
                    sell = True
                    self.Log(f"Time expired: {his.index[0][1].strftime('%Y-%m-%d %H:%M')}")
            
            if self.extendWhenSignalled:
                if (signal == BUY_SIGNAL and q > 0) or (signal == SELL_SIGNAL and q < 0):
                    self.placedTrade = self.Time.strftime("%Y-%m-%d %H:%M") #extend
                    if self.updateExePriceWhenExtend:
                        self.exePrice = price
                    self.Log(f"Time extended")
            
            if self.sellWhenSignalChange:
                if (signal == BUY_SIGNAL and q < 0) or (signal == SELL_SIGNAL and q > 0):
                    sell = True
                    self.Log(f"Signal Changed: sell")

            if not sell and self.sellatnpercent:
                self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
                if q>0:
                    if price > self.exePrice * (1 + self.npercent):
                        self.MarketOrder(self.stock, -q)
                    else:
                        self.LimitOrder(self.stock, -q, self.exePrice * (1 + self.npercent) , tag="take profit")
                elif q<0:
                    if price < self.exePrice * (1 - self.npercent):
                        self.MarketOrder(self.stock, -q)
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
                elif q<0:
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
            if signal == BUY_SIGNAL:
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.stock, q)
                self.Log(f"Price: {price}, signal: {signal}, buy {q}")
                self.traded = True
                self.placedTrade = self.Time.strftime("%Y-%m-%d %H:%M")
            elif signal == SELL_SIGNAL:
                q = self.Portfolio.Cash/price 
                ticket = self.MarketOrder(self.stock, -q)
                self.Log(f"Price: {price}, signal: {signal}, sell {q}")
                self.traded = True
                self.placedTrade = self.Time.strftime("%Y-%m-%d %H:%M")