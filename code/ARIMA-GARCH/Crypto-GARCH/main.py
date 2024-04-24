# region imports
from AlgorithmImports import *
# endregion
from ModelGARCH import GarchModel

# The trading strategy based on the GARCH model in the US Stock market. 
# NOTE: This algorithm is not published in the report and final results, 
# and it is only involved in the preliminary development process.
class GARCHCrypto(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. 
        All algorithms must be initialized before performing testing.'''
        # basic configs
        self.SetStartDate(2019, 9, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31)
        self.SetCash(1000000)  # Set Strategy Cash
        # self.Settings.FreePortfolioValuePercentage = 0.2 # set cash ratio
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.SetTimeZone(TimeZones.Toronto)

        # set a benchmark
        # self.SetBenchmark("SPY")

        # tickers
        ticker = self.AddCrypto("AVAXBUSD", Resolution.Daily, Market.Binance)
        self.ticker = ticker.Symbol
        ticker.SetDataNormalizationMode(DataNormalizationMode.Raw)

        # additional defined variables - data
        self.pastClosingPrices = None

        # additional defined variables - model
        self.daysBefore = 100
        self.model = None
        self.GARCHp = 2
        self.GARCHq = 1

        # additional defined variables - trade
        self.sellPositionsRatio = 0.75
        self.predictedLowMargin = 0.05
        self.predictedHighMargin = 0.1
        self.recentBuyPrice = 0
        self.recentSellPrice = np.inf
        self.takeProfitLevel = 1.7
        self.stopLossRatio = 0.95

    def OnData(self, data: Slice):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''

        # preliminary check of data availability, 
        # if data is not available, exit the function
        if not self.ticker in data:
            return
        if not data.Bars.ContainsKey(self.ticker):
            return
        
        # obtain the past closing prices
        self.pastClosingPrices = self.GetPastClosingPrices(self.daysBefore)
        if self.pastClosingPrices is None:
            return
        currentDayLow, currentDayHigh = data.Bars[self.ticker].Low, data.Bars[self.ticker].High

        # based on the price history, obtain the predicted quantities
        predicted, predictedStd = self.FitPredictModel()
        if predicted is None:
            return
        
        # obtains the current price, holding, average price and positions and the cash available
        currentPrice = data.Bars[self.ticker].Close
        holding = self.Portfolio[self.ticker]
        averagePrice = holding.AveragePrice
        positions = holding.Quantity

        # based on the triggering of buy and sell signals, trade accordingly
        if not self.Portfolio.Invested:
            pass
            # self.SetHoldings(self.ticker, 1)
        if self.BuySignalTriggered(currentDayLow, predicted):
            buyQuantity = self.Portfolio.Cash / currentPrice
            predictedLow = predicted * (1 - self.predictedLowMargin)
            buyPrice = round(predictedLow, 2)
            ticket = self.LimitOrder(self.ticker, buyQuantity, buyPrice)
            self.Log(f"Buy order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
        elif self.SellSignalTriggered(currentDayHigh, predicted, predictedStd, averagePrice):
            predictedHigh = predicted * (1 + self.predictedHighMargin)
            sellQuantity = -int(positions * self.sellPositionsRatio)
            sellPrice = round(predictedHigh, 2)
            ticket = self.LimitOrder(self.ticker, sellQuantity, sellPrice)
            self.Log(f"Sell order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")

    def GetPastClosingPrices(self, daysBefore: int) -> np.array:
        '''Obtains the past closing prices. 

        Arguments: 
            daysBefore: The number of days before the current time point. 

        Returns: A numpy array containing the historical closing prices for the past history requested. 
        '''
        pastPrices = []
        slices = self.History(self.ticker, daysBefore, Resolution.Daily)
        try:
            pastPrices = slices['close']
            return np.array(pastPrices)
        except KeyError:
            return None
    
    def FitPredictModel(self):
        '''Fits the GARCH model and obtain the predicted quantity. '''
        try:
            lastObsValue = self.pastClosingPrices[-1]
            self.model = GarchModel(self.pastClosingPrices, lastObsValue, p=self.GARCHp, q=self.GARCHq)
            self.model.fit()
            forecastMean, forecastStd = self.model.predict_forecasts()
            return forecastMean, forecastStd
        except ValueError:
            return None, None
    
    def BuySignalTriggered(self, currentLow: float, predictedLow: float) -> bool:
        '''Determines if the buy signal will be triggered or not. 
        
        Arguments: 
            currentLow: The current low price of the underlying. 
            predictedLow: The predicted low price of the underlying. 

        Returns: Whether the buy signal is triggered or not. 
        '''
        triggered = (currentLow < predictedLow) and (self.recentSellPrice > predictedLow)
        if triggered:
            self.Log(f"Buy signal triggered")
        return triggered
    
    def SellSignalTriggered(self, currentHigh: float, predictedHigh: float, volatility: float, averagePrice: float, 
                            stdDevLoading: float=1.5) -> bool:
        '''Determines if the sell signal will be triggered or not. 
        
        Arguments: 
            currentHigh: The current high price of the underlying. 
            predictedHigh: The predicted high price of the underlying. 
            averagePrice: The average price of the portfolio. 
            stdDevLoading: The standard deviation loading to incorporate in the sell signal. 

        Returns: Whether the sell signal is triggered or not. 
        '''
        priceStd = volatility
        triggered = (((currentHigh > predictedHigh) and \
            (self.recentBuyPrice + stdDevLoading * priceStd < predictedHigh)) and \
                averagePrice * self.takeProfitLevel < predictedHigh or \
                    (averagePrice * self.takeProfitLevel > currentHigh)) or \
                        (predictedHigh < averagePrice * self.stopLossRatio)
        if triggered:
            self.Log(f"Sell signal triggered")
        return triggered
