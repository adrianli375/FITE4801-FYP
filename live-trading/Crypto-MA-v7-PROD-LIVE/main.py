from AlgorithmImports import *
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import pandas as pd
import numpy as np
from keras.optimizers import SGD


class CryptoMA(QCAlgorithm):
    
    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2023,1,1)    #Set Start Date
        self.SetEndDate(2024,2,29)      #Set End Date
        self.SetAccountCurrency("USDT")
        self.SetCash("USDT",1080)           #Set Strategy Cash
        self.initialCash = 1080

        ### Instrument
        self.instrument = "BTCUSDT"

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
        self.volatility_n_days = 10
        self.past_volatility_n_days = 5
        self.volatility_coefficient = 0.6

        self.close_volatility_coefficient = 0.05

        #LSTM param
        self.resolution = 200
        self.retrain = 100

        #v7 param
        self.penalty_coefficient = 0.003
        self.num_days_lookback = 45
        self.adjustCloseVol = False
        ### End Parameters


        self.upperlinepos = ""
        self.lowerlinepos = ""
        self.cur_purchaseprice = 0
        self.cont_liquidate = False
        self.highwatermark = 0

        self.train = False

    def TrainAlgo(self):
        history = self.History(self.symbol, self.resolution, Resolution.Daily)
        if history.shape[0] != self.resolution:
            return
       
        open = history["open"].values
        high = history["high"].values
        low = history["low"].values
        close = history["close"].values
        volume = history["volume"].values
        x_train = []
        y_train = []
        for i in range(self.past_volatility_n_days,len(close)-self.volatility_n_days):
            x_train.append([open[i-self.past_volatility_n_days:i],high[i-self.past_volatility_n_days:i],low[i-self.past_volatility_n_days:i],close[i-self.past_volatility_n_days:i],volume[i-self.past_volatility_n_days:i]])
            y_train.append([pd.DataFrame({"Close":close[i:i+self.volatility_n_days]})["Close"].pct_change().dropna().std()])
        #Model

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        self.regressorLSTM = Sequential()
        # First LSMT layer
        self.regressorLSTM.add(LSTM(units=512, return_sequences=True, input_shape=(x_train.shape[1],self.past_volatility_n_days)))
        self.regressorLSTM.add(Dropout(0.2))
        # Second LSTM layer
        self.regressorLSTM.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1],self.past_volatility_n_days)))
        self.regressorLSTM.add(Dropout(0.2))
        # Third LSTM layer
        self.regressorLSTM.add(LSTM(units=128, return_sequences=False, input_shape=(x_train.shape[1],self.past_volatility_n_days)))
        self.regressorLSTM.add(Dropout(0.2))
        # The output layer
        self.regressorLSTM.add(Dense(units=1))
        # Compiling the RNN
        self.regressorLSTM.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=False),loss='mean_squared_error')
        # Fitting to the training set
        self.regressorLSTM.fit(x_train,y_train,epochs=50,validation_split=0.1,batch_size=150)

        self.train = True
        self.lasttraintime = self.Time
        return


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

        if self.train and self.lasttraintime + timedelta(days=self.retrain) < self.Time:
            self.train = False
        
        if not self.train:
            self.TrainAlgo()
            if not self.train:
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

        df3 = self.History(self.symbol, self.past_volatility_n_days, Resolution.Daily)
        open = df3["open"].values
        high = df3["high"].values
        low = df3["low"].values
        close = df3["close"].values
        volume = df3["volume"].values

        x_test = np.array([[open,high,low,close,volume]])
        try:
            y_predict = self.regressorLSTM.predict(x_test)
        except:
            return
        # y_predict = np.squeeze(y_predict)
        # self.Log(y_predict)
        ### (v7) Past trades control
        trades = self.TradeBuilder.ClosedTrades
        trades = trades[-min(len(trades),self.num_days_lookback*10):]
        pnl_count = 0 #+ve: loss
        for trade in trades:
            if trade.ExitTime > self.UtcTime - timedelta(days=self.num_days_lookback) and trade.Quantity*price >= self.initialCash//2:
                if not trade.IsWin:
                    pnl_count += 1
                else:
                    pnl_count -= 1
                
        pnl_count = max(pnl_count,0)
        ### End (v7)

        self.percent_above = y_predict[0] * self.volatility_coefficient + pnl_count * self.penalty_coefficient
        if self.adjustCloseVol:
            self.close_above = y_predict[0] * self.close_volatility_coefficient + pnl_count * self.penalty_coefficient
        else:
            self.close_above = y_predict[0] * self.close_volatility_coefficient

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
