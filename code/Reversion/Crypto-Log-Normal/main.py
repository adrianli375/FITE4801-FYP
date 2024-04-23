# region imports
from AlgorithmImports import *
# endregion
from lognormal import LogNormal

# The trading strategy based on the log-normal assumption in the cryptocurrency market. 
# NOTE: This algorithm is not published in the report and final results, 
# and it is only involved in the preliminary development process. 
class VaRTrading(QCAlgorithm):

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
        self.SetBenchmark("SPY")

        # tickers
        ticker = self.AddCrypto('BTCUSD', Resolution.Minute, Market.Bitfinex)
        self.ticker = ticker.Symbol
        ticker.SetDataNormalizationMode(DataNormalizationMode.Raw)

        # additional defined variables - data
        self.pastClosingPrices = None

        # additional defined variables - model
        self.minutesBefore = 30
        self.model = None
        self.confLevel = 0.9

        # additional defined variables - trade
        self.recentBuyPrice = 0
        self.recentSellPrice = np.inf
        self.stopLossRatio = 0.05


    def OnData(self, data: Slice):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''

        # first, check the existence of the data
        # if the data requested does not exist, exit the function. 
        if not self.ticker in data:
            return
        if not data.Bars.ContainsKey(self.ticker):
            return
        
        # obtain high, low, close prices
        self.pastClosingPrices = self.GetPastClosingPrices(self.minutesBefore)
        currentDayLow, currentDayHigh = data.Bars[self.ticker].Low, data.Bars[self.ticker].High
        holding = self.Portfolio[self.ticker]

        # obtain the current cash available, the average price of the portfolio, 
        # the current price of the stock and the current positions holding. 
        cash = self.Portfolio.Cash
        averagePrice = holding.AveragePrice
        currentPrice = data.Bars[self.ticker].Close
        positions = holding.Quantity
        
        if len(self.pastClosingPrices) < 2:
            return

        # fit model
        self.model = LogNormal(self.pastClosingPrices)

        # trade
        if (positions > 0 and currentPrice < averagePrice * (1 - self.stopLossRatio)) or (positions < 0 and currentPrice > (1 + self.stopLossRatio)):
            self.Liquidate()
        elif self.BuySignalTriggered(currentDayLow):
            quantity = cash / currentPrice
            price = self.CalculateVaR(1 - self.confLevel)
            ticket = self.LimitOrder(self.ticker, quantity, price)
            self.Log(f"Buy order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
        elif self.SellSignalTriggered(currentDayHigh):
            if positions != 0:
                quantity = -positions
            else:
                quantity = -cash / currentPrice
            price = self.CalculateVaR(self.confLevel)
            ticket = self.LimitOrder(self.ticker, quantity, price)
            self.Log(f"Buy order: Quantity filled: {ticket.QuantityFilled}; Fill price: {ticket.AverageFillPrice}")
    
    def OnOrderEvent(self, orderEvent: OrderEvent):
        '''OnOrder event is an entry point which processes an order. 
        Arguments:
            orderEvent: OrderEvent object containing the order details. 
        '''

        # updates the most recent buy and sell price based on the order
        if orderEvent.Status == OrderStatus.Filled or orderEvent.Status == OrderStatus.PartiallyFilled:
            isBuyOrder = orderEvent.FillQuantity > 0
            if isBuyOrder:
                self.recentBuyPrice = orderEvent.FillPrice
            else:
                self.recentSellPrice = orderEvent.FillPrice
    
    def GetPastClosingPrices(self, minutesBefore: int) -> np.array:
        '''Obtains the past closing prices. 

        Arguments: 
            minutesBefore: The number of minutes before the current time point. 

        Returns: A numpy array containing the historical closing prices for the past history requested. 
        '''
        pastPrices = []
        slices = self.History(minutesBefore)
        for s in slices:
            if s.Bars.ContainsKey(self.ticker):
                closingPrice = s.Bars[self.ticker].Close
                pastPrices.append(closingPrice)
        return np.array(pastPrices)
    
    def CalculateVaR(self, alpha: float) -> float:
        '''Calculates the value at risk at a given probability level. 
        
        Arguments: 
            alpha: The probability level. 
        
        Returns: The calculated value at risk (percentile). 
        '''
        return self.model.calculate_value_at_risk(alpha)
    
    def BuySignalTriggered(self, currentDayLow: float) -> bool:
        '''Determines if the buy signal is triggered. 
        
        Arguments:
            currentDayLow: The low price of the underlying. 
        
        Returns: A boolean value indicating whether the buy signal is triggered. 
        '''
        var = self.CalculateVaR(1 - self.confLevel)
        triggered = currentDayLow < var and self.recentSellPrice > var
        if triggered:
            self.Log("Buy signal triggered")
        return triggered
    
    def SellSignalTriggered(self, currentDayHigh: float) -> bool:
        '''Determines if the sell signal is triggered. 
        
        Arguments:
            currentDayHigh: The high price of the underlying. 
        
        Returns: A boolean value indicating whether the sell signal is triggered. 
        '''
        var = self.CalculateVaR(self.confLevel)
        triggered = currentDayHigh > var and self.recentBuyPrice < var
        if triggered:
            self.Log("Sell signal triggered")
        return triggered
