# region imports
from AlgorithmImports import *
# endregion
from ModelARIMA import ArimaModel

class ARIMACrypto(QCAlgorithm):

    def Initialize(self):
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
        ticker = self.AddCrypto("ETHBUSD", Resolution.Daily)
        self.ticker = ticker.Symbol
        ticker.SetDataNormalizationMode(DataNormalizationMode.Raw)

        # additional defined variables - data
        self.pastClosingPrices = None

        # additional defined variables - model
        self.daysBefore = 250
        self.model = None
        self.modelTSOrder = (1, 1, 0) # corresponding to (p, d, q) for the class of ARIMA models
        self.predictionIntervalConfidenceLevel = 0.95

        # additional defined variables - trade
        self.buyQuantity = 50
        self.sellPositionsRatio = 0.75
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
        if self.pastClosingPrices is None:
            return
        currentDayLow, currentDayHigh = data.Bars[self.ticker].Low, data.Bars[self.ticker].High
        predictedLow, predicted, predictedHigh = self.FitPredictModel()
        if predictedLow is None or predicted is None or predictedHigh is None:
            return
        holding = self.Portfolio[self.ticker]
        averagePrice = holding.AveragePrice
        positions = holding.Quantity

        if not self.Portfolio.Invested:
            self.SetHoldings(self.ticker, 1)
        elif self.BuySignalTriggered(currentDayLow, predictedLow):
            buyPrice = round(predictedLow, 2)
            ticket = self.LimitOrder(self.ticker, self.buyQuantity, buyPrice)
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
        slices = self.History(self.ticker, daysBefore, Resolution.Daily)
        try:
            pastPrices = slices['close']
            return np.array(pastPrices)
        except KeyError:
            return None
    
    def FitPredictModel(self) -> (int, int, int):
        if len(self.pastClosingPrices) <= 2:
            return None, None, None
        try:
            self.model = ArimaModel(self.pastClosingPrices, order=self.modelTSOrder, 
                                    percent_ci=self.predictionIntervalConfidenceLevel)
            self.model.fit()
            lower, est, upper = self.model.predict_forecasts()
            return lower, est, upper
        except Exception as e:
            return None, None, None
    
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