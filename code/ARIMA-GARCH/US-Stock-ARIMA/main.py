# region imports
from AlgorithmImports import *
# endregion
from ModelARIMA import ArimaModel

class ARIMA1(QCAlgorithm):

    def Initialize(self):
        # basic configs
        self.SetStartDate(2013, 1, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31)
        self.SetCash(1000000)  # Set Strategy Cash
        # self.Settings.FreePortfolioValuePercentage = 0.2 # set cash ratio
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetTimeZone(TimeZones.Toronto)

        # set a benchmark
        self.SetBenchmark("SPY")

        # tickers
        ticker = self.AddEquity("UNH", Resolution.Daily)
        self.ticker = ticker.Symbol
        ticker.SetDataNormalizationMode(DataNormalizationMode.Raw)

        # additional defined variables - data
        self.pastClosingPrices = None

        # additional defined variables - model
        self.daysBefore = 250
        self.model = None
        self.modelTSOrder = (1, 1, 0) # corresponding to (p, d, q) for the class of ARIMA models
        self.predictionIntervalConfidenceLevel = 0.9

        # additional defined variables - trade
        self.sellPositionsRatio = 0.75
        self.recentBuyPrice = 0
        self.recentSellPrice = np.inf
        self.takeProfitLevel = 1.3
        self.stopLossRatio = 0.95

    def OnData(self, data: Slice):
        if not self.ticker in data:
            return
        if not data.Bars.ContainsKey(self.ticker):
            return
        self.pastClosingPrices = self.GetPastClosingPrices(self.daysBefore)
        currentDayLow, currentDayHigh = data.Bars[self.ticker].Low, data.Bars[self.ticker].High
        predictedLow, predicted, predictedHigh = self.FitPredictModel()
        holding = self.Portfolio[self.ticker]
        averagePrice = holding.AveragePrice
        positions = holding.Quantity
        cashAvailable = self.Portfolio.CashBook["USD"].Amount

        if not self.Portfolio.Invested:
            self.SetHoldings(self.ticker, 1)
        elif self.BuySignalTriggered(currentDayLow, predictedLow):
            buyPrice = round(predictedLow, 2)
            buyQuantity = max(0, cashAvailable / currentDayHigh)
            ticket = self.LimitOrder(self.ticker, buyQuantity, buyPrice)
            self.Log(f"Buy order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
        elif (self.SellSignalTriggered(currentDayHigh, predictedHigh, averagePrice) and positions > 0) or averagePrice > currentDayLow * self.stopLossRatio:
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
    
    def FitPredictModel(self) -> (int, int, int):
        self.model = ArimaModel(self.pastClosingPrices, order=self.modelTSOrder, 
                                percent_ci=self.predictionIntervalConfidenceLevel)
        self.model.fit()
        lower, est, upper = self.model.predict_forecasts()
        return lower, est, upper
    
    def BuySignalTriggered(self, currentLow: float, predictedLow: float) -> bool:
        triggered = (currentLow < predictedLow) and (self.recentSellPrice > predictedLow)
        if triggered:
            self.Log(f"Buy signal triggered")
        return triggered
    
    def SellSignalTriggered(self, currentHigh: float, predictedHigh: float, averagePrice: float, 
                            stdDevLoading: float=1.5) -> bool:
        priceStd = self.pastClosingPrices.std()
        triggered = (((currentHigh > predictedHigh) and \
            (self.recentBuyPrice + stdDevLoading * priceStd < predictedHigh)) and \
                averagePrice * self.takeProfitLevel < predictedHigh or \
                    (averagePrice * self.takeProfitLevel > currentHigh))
        if triggered:
            self.Log(f"Sell signal triggered")
        return triggered