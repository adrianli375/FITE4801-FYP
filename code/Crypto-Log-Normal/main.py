# region imports
from AlgorithmImports import *
# endregion
from lognormal import LogNormal

class VaRTrading(QCAlgorithm):

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
        ticker = self.AddCrypto('BTCBUSD', Resolution.Daily, Market.Binance)
        self.ticker = ticker.Symbol
        ticker.SetDataNormalizationMode(DataNormalizationMode.Raw)

        # additional defined variables - data
        self.pastClosingPrices = None

        # additional defined variables - model
        self.daysBefore = 50
        self.model = None
        self.confLevel = 0.8

        # additional defined variables - trade
        self.sellPositionsRatio = 0.75
        self.strongSellPositionsRatio = 0.9
        self.recentBuyPrice = 0
        self.recentSellPrice = np.inf


    def OnData(self, data: Slice):
        if not self.ticker in data:
            return
        if not data.Bars.ContainsKey(self.ticker):
            return
        self.pastClosingPrices = self.GetPastClosingPrices(self.daysBefore)
        currentDayLow, currentDayHigh = data.Bars[self.ticker].Low, data.Bars[self.ticker].High
        holding = self.Portfolio[self.ticker]
        cash = self.Portfolio.Cash
        averagePrice = holding.AveragePrice
        currentPrice = data.Bars[self.ticker].Close
        positions = holding.Quantity
        buyQuantity = cash / currentPrice
        
        if len(self.pastClosingPrices) < 2:
            return

        # fit model
        self.model = LogNormal(self.pastClosingPrices)

        # trade
        if self.StrongSellSignalTriggered(currentDayHigh):
            quantity = -int(positions * self.strongSellPositionsRatio)
            price = self.CalculateTVaR(self.confLevel)
            ticket = self.LimitOrder(self.ticker, quantity, price)
            self.Log(f"Sell order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
        elif self.BuySignalTriggered(currentDayLow):
            quantity = buyQuantity
            price = self.CalculateVaR(1 - self.confLevel)
            ticket = self.LimitOrder(self.ticker, quantity, price)
            self.Log(f"Buy order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
        elif self.SellSignalTriggered(currentDayHigh):
            quantity = -int(positions * self.sellPositionsRatio)
            price = self.CalculateVaR(self.confLevel)
            ticket = self.LimitOrder(self.ticker, quantity, price)
            self.Log(f"Buy order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
    
    def OnOrderEvent(self, orderEvent: OrderEvent):
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if orderEvent.Status == OrderStatus.Filled or orderEvent.Status == OrderStatus.PartiallyFilled:
            isBuyOrder = orderEvent.FillQuantity > 0
            if isBuyOrder:
                self.recentBuyPrice = orderEvent.FillPrice
            else:
                self.recentSellPrice = orderEvent.FillPrice
    
    def GetPastClosingPrices(self, daysBefore: int) -> np.array:
        pastPrices = []
        slices = self.History(daysBefore)
        for s in slices:
            if s.Bars.ContainsKey(self.ticker):
                closingPrice = s.Bars[self.ticker].Close
                pastPrices.append(closingPrice)
        return np.array(pastPrices)
    
    def CalculateVaR(self, alpha: float):
        return self.model.calculate_value_at_risk(alpha)
    
    def CalculateTVaR(self, alpha: float):
        return self.model.calculate_tail_value_at_risk(alpha)
    
    def BuySignalTriggered(self, currentDayLow: float):
        var = self.CalculateVaR(1 - self.confLevel)
        triggered = currentDayLow < var and self.recentSellPrice > var
        if triggered:
            self.Log("Buy signal triggered")
        return triggered
    
    def SellSignalTriggered(self, currentDayHigh: float):
        var = self.CalculateVaR(self.confLevel)
        triggered = currentDayHigh > var and self.recentBuyPrice < var
        if triggered:
            self.Log("Sell signal triggered")
        return triggered
    
    def StrongSellSignalTriggered(self, currentDayHigh: float):
        tvar = self.CalculateTVaR(self.confLevel)
        triggered = currentDayHigh > tvar and self.recentBuyPrice < tvar
        if triggered:
            self.Log("Strong sell signal triggered")
        return triggered

