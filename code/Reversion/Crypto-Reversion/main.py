# region imports
from AlgorithmImports import *
# endregion
import numpy as numpy
from scipy.stats import t
import talib as tb


# The reversion trading strategy in the cryptocurrency market. 
class CryptoReversion(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. 
        All algorithms must be initialized before performing testing.'''
        self.SetStartDate(2019, 9, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(1000000)

        # params
        self.setResolution = Resolution.Daily
        self.takeProfitPercent = None
        self.stopLossPercent = None
        self.nDaysMovingAverage = 5
        self.nDaysHistory = 365
        self.percentile = 0.05
        self.extremePercentile = 0.01

        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)

        self.ticker = 'BTCBUSD'
        self.stock = self.AddCrypto(self.ticker, self.setResolution, Market.Binance).Symbol
        self.stoppedTrading = False

    def OnData(self, data: Slice):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''

        # first, check the existence of the data
        # if the data requested does not exist, exit the function. 
        if self.stock in data.Bars:
            tradeBar = data.Bars[self.stock]
            currentPrice = tradeBar.Close
        else:
            return
        
        # find the average prices, position and quantity in the existing portfolio
        avgPrice = self.Portfolio[self.stock].AveragePrice
        positions = self.Portfolio[self.stock].Quantity
        quantity = int(self.Portfolio.Cash / currentPrice)

        # obtain the past history of the stock
        history = self.History(self.stock, self.nDaysHistory + self.nDaysMovingAverage, self.setResolution)

        # if the history obtain is less than the moving average length, early exit the function
        if len(history) <= self.nDaysMovingAverage:
            return
        
        # calculate the price to MA ratios in the history
        history['MA'] = tb.MA(history['close'], timeperiod=self.nDaysMovingAverage)
        history['Ratio'] = history['close'] / history['MA'] # price to MA ratio

        # drop empty values as they are not used for calculation
        history.dropna(inplace=True)

        # fit the distribution to the past price-to-MA ratios
        pastRatios = history['Ratio'].values[:-1]
        currentRatio = history['Ratio'].values[-1]

        # determine the ratios to long, short and stop loss based on the critical values
        lowRatio, highRatio = self.getLowAndHighRatios(pastRatios, self.percentile)
        stopTradingLowRatio, stopTradingHighRatio = self.getLowAndHighRatios(pastRatios, self.extremePercentile)
        self.stopLossPercent = 1 - pastRatios.min()
        self.takeProfitPercent = pastRatios.max() - 1

        # adjust the stoppedTrading variable accordingly
        if currentRatio < stopTradingLowRatio or currentRatio > stopTradingHighRatio:
            self.stoppedTrading = True
        else:
            self.stoppedTrading = False

        currentDate = self.Time.strftime("%Y-%m-%d")

        # trade accordingly based on the current obtained price-to-MA ratio
        # if the ratio obtained is lower than a threshold, trigger a buy / short buy signal
        # if the ratio obtained is higher than a threshold, trigger a (short) sell signal
        if not self.stoppedTrading:
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
        '''Method to obtain the upper and lower critical values of the price-to-MA ratios. 
        
        Arguments:
            x: The input price array
            percentile: The confidence interval percentage. 
        
        Returns: The lower critical value and upper critical values of the fitted distribution. 
        '''
        degFreedom, mean, sd = t.fit(x)
        lowerCriticalValue = t.ppf(percentile / 2, degFreedom, mean, sd)
        upperCriticalValue = t.ppf(1 - percentile / 2, degFreedom, mean, sd)
        return lowerCriticalValue, upperCriticalValue