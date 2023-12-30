# region imports
from AlgorithmImports import *
# endregion
import numpy as numpy
from scipy.stats import t
import talib as tb

class MuscularGreenSardine(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2019, 9, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(1000000)

        # params
        self.setResolution = Resolution.Daily
        self.takeProfitPercent = 0.2
        self.stopLossPercent = 0.075
        self.nDaysMovingAverage = 25
        self.nDaysHistory = 365
        self.percentile = 0.05

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)

        self.ticker = 'BTCBUSD'
        self.stock = self.AddCrypto(self.ticker, self.setResolution, Market.Binance).Symbol

    def OnData(self, data: Slice):
        if self.stock in data.Bars:
            tradeBar = data.Bars[self.stock]
            currentPrice = tradeBar.Close
        else:
            return
        
        avgPrice = self.Portfolio[self.stock].AveragePrice
        positions = self.Portfolio[self.stock].Quantity
        quantity = int(self.Portfolio.Cash / currentPrice)

        history = self.History(self.stock, self.nDaysHistory + self.nDaysMovingAverage, self.setResolution)
        if len(history) <= self.nDaysMovingAverage:
            return
        history['MA'] = tb.MA(history['close'], timeperiod=self.nDaysMovingAverage)
        history['Ratio'] = history['close'] / history['MA'] # price to MA ratio

        history.dropna(inplace=True)

        pastRatios = history['Ratio'].values[:-1]
        currentRatio = history['Ratio'].values[-1]
        lowRatio, highRatio = self.getLowAndHighRatios(pastRatios, self.percentile)

        currentDate = self.Time.strftime("%Y-%m-%d")

        if (positions > 0 and avgPrice * (1 - self.stopLossPercent) > currentPrice) or \
            (positions < 0 and avgPrice * (1 + self.stopLossPercent) < currentPrice):
            self.Liquidate()
        if currentRatio < lowRatio:
            self.Log(f'{currentDate}, Buy signal triggered, positions = {positions}')
            if positions >= 0:
                self.StopLimitOrder(self.stock, quantity, currentPrice * (1 - self.stopLossPercent), currentPrice)
            elif positions < 0 and avgPrice * (1 - self.takeProfitPercent) > currentPrice:
                self.MarketOrder(self.stock, -positions)
        elif currentRatio > highRatio:
            self.Log(f'{currentDate}, Sell signal triggered, positions = {positions}')
            if positions <= 0:
                self.StopLimitOrder(self.stock, -quantity, currentPrice * (1 + self.stopLossPercent), currentPrice)
            elif positions > 0 and avgPrice * (1 + self.takeProfitPercent) < currentPrice:
                self.MarketOrder(self.stock, -positions)

    
    def getLowAndHighRatios(self, x: np.array, percentile: float) -> (float, float):
        degFreedom, mean, sd = t.fit(x)
        lowerCriticalValue = t.ppf(percentile / 2, degFreedom, mean, sd)
        upperCriticalValue = t.ppf(1 - percentile / 2, degFreedom, mean, sd)
        return lowerCriticalValue, upperCriticalValue