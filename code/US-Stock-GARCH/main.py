# region imports
from AlgorithmImports import *
# endregion
from ModelGARCH import GarchModel

class GARCH1(QCAlgorithm):

    def Initialize(self):
        # basic configs
        self.SetStartDate(2013, 1, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31)
        # self.SetAccountCurrency("HKD")
        self.SetCash(100000)  # Set Strategy Cash
        self.Settings.FreePortfolioValuePercentage = 0.2 # set cash ratio
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetTimeZone(TimeZones.Toronto)

        # set a benchmark
        self.SetBenchmark("SPY")

        # tickers
        ticker = self.AddEquity("XOM", Resolution.Daily)
        self.ticker = ticker.Symbol
        ticker.SetDataNormalizationMode(DataNormalizationMode.Raw)

        # additional defined variables - data
        self.pastClosingPrices = None

        # additional defined variables - model
        self.daysBefore = 250
        self.model = None
        self.GARCHp = 2
        self.GARCHq = 2

        # additional defined variables - trade
        self.buyQuantity = 50
        self.sellPositionsRatio = 0.75
        self.predictedLowMargin = 0.05
        self.predictedHighMargin = 0.1
        self.recentBuyPrice = 0
        self.recentSellPrice = np.inf
        self.takeProfitLevel = 1.5
        self.stopLossRatio = 0.9

    def OnData(self, data: Slice):
        if not self.ticker in data:
            return
        if not data.Bars.ContainsKey(self.ticker):
            return
        self.pastClosingPrices = self.GetPastClosingPrices(self.daysBefore)
        currentDayLow, currentDayHigh = data.Bars[self.ticker].Low, data.Bars[self.ticker].High
        predicted, predictedStd = self.FitPredictModel()
        holding = self.Portfolio[self.ticker]
        averagePrice = holding.AveragePrice
        positions = holding.Quantity

        if not self.Portfolio.Invested:
            pass
            # self.SetHoldings(self.ticker, 1)
        if self.BuySignalTriggered(currentDayLow, predicted):
            predictedLow = predicted * (1 - self.predictedLowMargin)
            buyPrice = round(predictedLow, 2)
            ticket = self.LimitOrder(self.ticker, self.buyQuantity, buyPrice)
            self.Log(f"Buy order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
        elif self.SellSignalTriggered(currentDayHigh, predicted, predictedStd, averagePrice):
            predictedHigh = predicted * (1 + self.predictedHighMargin)
            sellQuantity = -int(positions * self.sellPositionsRatio)
            sellPrice = round(predictedHigh, 2)
            ticket = self.LimitOrder(self.ticker, sellQuantity, sellPrice)
            self.Log(f"Sell order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
    
    def OnOrderEvent(self, orderEvent):
        pass

    def GetPastClosingPrices(self, daysBefore: int) -> np.array:
        pastPrices = []
        slices = self.History(daysBefore)
        for s in slices:
            closingPrice = s.Bars[self.ticker].Close
            pastPrices.append(closingPrice)
        return np.array(pastPrices)
    
    def FitPredictModel(self):
        lastObsValue = self.pastClosingPrices[-1]
        self.model = GarchModel(self.pastClosingPrices, lastObsValue, p=self.GARCHp, q=self.GARCHq)
        self.model.fit()
        forecastMean, forecastStd = self.model.predict_forecasts()
        return forecastMean, forecastStd
    
    def BuySignalTriggered(self, currentLow: float, predictedLow: float) -> bool:
        triggered = (currentLow < predictedLow) and (self.recentSellPrice > predictedLow)
        if triggered:
            self.Log(f"Buy signal triggered")
        return triggered
    
    def SellSignalTriggered(self, currentHigh: float, predictedHigh: float, volatility: float, averagePrice: float, 
                            stdDevLoading: float=1.5) -> bool:
        priceStd = volatility
        triggered = (((currentHigh > predictedHigh) and \
            (self.recentBuyPrice + stdDevLoading * priceStd < predictedHigh)) and \
                averagePrice * self.takeProfitLevel < predictedHigh or \
                    (averagePrice * self.takeProfitLevel > currentHigh)) or \
                        (predictedHigh < averagePrice * self.stopLossRatio)
        if triggered:
            self.Log(f"Sell signal triggered")
        return triggered
