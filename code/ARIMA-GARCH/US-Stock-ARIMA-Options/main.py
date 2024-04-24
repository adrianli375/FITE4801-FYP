# region imports
from AlgorithmImports import *
# endregion
from ModelARIMA import ArimaModel
from typing import Union


# The trading strategy based on the ARIMA model and options trading in the US Stock market. 
# NOTE: This algorithm is not published in the report and final results, 
# and it is only involved in the preliminary development process.
class ARIMAOptions(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. 
        All algorithms must be initialized before performing testing.'''
        # basic configs
        self.SetStartDate(2013, 1, 1)  # Set Start Date
        self.SetEndDate(2022, 12, 31)
        self.SetCash(1000000)  # Set Strategy Cash
        # self.Settings.FreePortfolioValuePercentage = 0.2 # set cash ratio
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetTimeZone(TimeZones.Toronto)
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw

        # set a benchmark
        self.SetBenchmark("SPY")

        # tickers
        ticker = self.AddEquity("SPY", Resolution.Daily)
        self.ticker = ticker.Symbol
        ticker.SetDataNormalizationMode(DataNormalizationMode.Raw)
        option = self.AddOption(self.ticker, Resolution.Daily)

        # additional defined variables - data
        self.pastClosingPrices = None

        # additional defined variables - model
        self.daysBefore = 250
        self.model = None
        self.modelTSOrder = (1, 1, 1) # corresponding to (p, d, q) for the class of ARIMA models
        self.predictionIntervalConfidenceLevel = 0.99

        # additional defined variables - trade
        self.predictNDays = 25
        self.stopLossPercent = 0.05

        # trade variables initialized
        self.buffer = 0.5
        self.state = 0
        self.lastCallStrikePrice = None
        self.lastPutStrikePrice = None
        self.sellDate = None

        # option specific params
        self.minExpiryDays = 25
        self.maxExpiryDays = 30
        option.SetFilter(-15, 15, timedelta(self.minExpiryDays), timedelta(self.maxExpiryDays))
        self.optionInvested = False

    def OnData(self, data: Slice):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''

        # preliminary check of data availability, 
        # if data is not available, exit the function
        if not self.ticker in data:
            return
        if not data.ContainsKey(self.ticker):
            return
        if not data.Bars.ContainsKey(self.ticker):
            return
        
        # 1. fit ARIMA(1, 1, 1) model daily to the past prices of the underlying
        self.pastClosingPrices = self.GetPastClosingPrices(self.daysBefore)
        if self.pastClosingPrices is None:
            return
        currentDayLow, currentDayHigh = data.Bars[self.ticker].Low, data.Bars[self.ticker].High
        currentPrice = data.Bars[self.ticker].Close
        predictedLow, predicted, predictedHigh = self.FitPredictModel()

        # 2. get the holding, price, positions and cash available
        holding = self.Portfolio[self.ticker]
        averagePrice = holding.AveragePrice
        positions = holding.Quantity
        cashAvailable = self.Portfolio.CashBook["USD"].Amount
        underlyingInvested = self.underlyingInvested()
        optionsInvested = self.optionsInvested()
        # stop loss
        if optionsInvested:
            sellOptionBeforeExpire = False
            # if current price is x% away from strike price and expiration is within 3 days, liquidate
            if self.sellDate is not None:
                if self.sellDate - timedelta(days=3) <= self.Time and self.sellDate >= self.Time:
                    if self.lastCallStrikePrice is not None and self.lastPutStrikePrice is not None:
                        if currentPrice > self.lastCallStrikePrice * (1 + self.stopLossPercent) or \
                            currentPrice < self.lastPutStrikePrice * (1 - self.stopLossPercent):
                            self.Liquidate()
                            self.lastCallStrikePrice, self.lastPutStrikePrice = None, None
                            self.state = 0
                    elif self.lastCallStrikePrice is not None:
                        if currentPrice > self.lastCallStrikePrice * (1 + self.stopLossPercent):
                            self.Liquidate()
                            self.lastCallStrikePrice, self.lastPutStrikePrice = None, None
                            self.state = 0
                    elif self.lastPutStrikePrice is not None:
                        if currentPrice < self.lastPutStrikePrice * (1 - self.stopLossPercent):
                            self.Liquidate()
                            self.lastCallStrikePrice, self.lastPutStrikePrice = None, None
                            self.state = 0
        if underlyingInvested:
            if positions < 100:
                self.Liquidate()
        
        # 3. from the predicted values, get the options that are closest to the predicted value and upper/lower bound
        # based on different rules, trade with the corresponding options involved
        # updated trading strategy based on https://slashtraders.com/en/blog/sp500-spy-etf-wheel-strategy/
        if not underlyingInvested and not optionsInvested and self.state == 0:
            try:
                chain = data.OptionChains.values()[0]
            except IndexError:
                return
            if not chain:
                return
            optionToShort = self.getPut(chain, predictedLow)
            if optionToShort is None:
                return
            strategyCost = optionToShort.LastPrice * 100
            try:
                quantity = int(abs(cashAvailable / strategyCost) / 100 * self.buffer)
            except ZeroDivisionError:
                return
            # short put -> state 0
            self.Sell(optionToShort.Symbol, quantity)
            self.lastCallStrikePrice = None
            self.lastPutStrikePrice = optionToShort.Strike
            self.sellDate = self.Time + timedelta(days=self.predictNDays)
            self.optionInvested = True
        elif not optionsInvested and underlyingInvested and self.state == 0:
            self.state = 1
        elif not optionsInvested and underlyingInvested and self.state == 1:
            try:
                chain = data.OptionChains.values()[0]
            except IndexError:
                return
            if not chain:
                return
            quantity = int(positions / 100)
            callOptionToShort = self.getCall(chain, predictedHigh)
            putOptionToShort = self.getPut(chain, predictedLow)
            if callOptionToShort is None or putOptionToShort is None or callOptionToShort.Strike == putOptionToShort.Strike:
                return
            # short call at predictedHigh and short put at predictedLow -> state 1
            self.Sell(callOptionToShort.Symbol, quantity)
            self.Sell(putOptionToShort.Symbol, quantity)
            self.lastCallStrikePrice = callOptionToShort.Strike
            self.lastPutStrikePrice = putOptionToShort.Strike
            self.sellDate = self.Time + timedelta(days=self.predictNDays)
        elif not optionsInvested and not underlyingInvested and self.state == 1:
            self.state = 0
        elif not optionsInvested and underlyingInvested and self.state == 1:
            self.state = 2
        elif underlyingInvested and not optionsInvested and self.state == 2:
            try:
                chain = data.OptionChains.values()[0]
            except IndexError:
                return
            if not chain:
                return
            optionToShort = self.getCall(chain, predictedHigh)
            if optionToShort is None:
                return
            quantity = int(positions / 100)
            # short call -> state 2
            self.Sell(optionToShort.Symbol, quantity)
            self.lastCallStrikePrice = callOptionToShort.Strike
            self.lastPutStrikePrice = None
            self.sellDate = self.Time + timedelta(days=self.predictNDays)
        elif not underlyingInvested and not optionsInvested and self.state == 2:
            self.state = 0
        self.Log(str(self.Time) + f'options invested: {optionsInvested}' + f'underlying invested: {underlyingInvested}' + f'state: {self.state}')

    def GetPastClosingPrices(self, daysBefore: int) -> Union[np.array, None]:
        '''Obtains the past closing prices. 

        Arguments: 
            daysBefore: The number of days before the current time point. 

        Returns: A numpy array containing the historical closing prices for the past history requested. 
        '''
        pastPrices_df = self.History([self.ticker], daysBefore)
        if len(pastPrices_df) == 0:
            return None
        pastPrices = pastPrices_df['close'].values
        return np.array(pastPrices)
    
    def FitPredictModel(self) -> (int, int, int):
        '''Fits the ARIMA model and obtain the predicted quantity. '''
        self.model = ArimaModel(self.pastClosingPrices, order=self.modelTSOrder, 
                                percent_ci=self.predictionIntervalConfidenceLevel)
        self.model.fit()
        lower, est, upper = self.model.predict_forecasts(n_steps=self.predictNDays)
        return lower, est, upper

    def underlyingInvested(self) -> bool:
        '''Determines if the underlying stock is invested or not. 
        
        Returns: Whether the underlying stock is invested or not. 
        '''
        holding = self.Portfolio[self.ticker]
        return holding.Invested
    
    def optionsInvested(self) -> bool:
        '''Determines if options are invested in the portfolio or not. 
        
        Returns: whether the options are invested in the portfolio or not. '''
        invested = False
        for symbol in self.Portfolio.keys():
            # self.Debug(f"Symbol: {symbol}")
            holding = self.Portfolio[symbol]
            securityType = holding.Type
            if securityType == 2: # check if the security type is an option
                if holding.Invested:
                    invested = True
        self.optionInvested = invested
        return invested
    
    def getCall(self, chain: OptionChain, price: float):
        '''Obtains the call option contract given the option chain and strike price. 
        
        Arguments:
            chain: The option chain. 
            price: The strike price. 

        Returns: The corresponding call option contract
        '''
        # Get the ATM strike price
        atm_strike = sorted(chain, key = lambda x: abs(price - x.Strike))[0].Strike
        # Select the ATM call Option contracts
        calls = [x for x in chain if x.Strike == atm_strike and x.Right == OptionRight.Call] 
        if len(calls) == 0: 
            return None
        # Select the contract with the furthest strike price
        contracts = sorted(calls, key=lambda x: abs(price - x.Strike))
        contract = contracts[0]
        return contract
    
    def getPut(self, chain: OptionChain, price: float):
        '''Obtains the put option contract given the option chain and strike price. 
        
        Arguments:
            chain: The option chain. 
            price: The strike price. 

        Returns: The corresponding put option contract
        '''
        # Get the ATM strike price
        atm_strike = sorted(chain, key = lambda x: abs(price - x.Strike))[0].Strike
        # Select the ATM call Option contracts
        puts = [x for x in chain if x.Strike == atm_strike and x.Right == OptionRight.Put] 
        if len(puts) == 0: 
            return None
        # Select the contract with the furthest strike price
        contracts = sorted(puts, key=lambda x: abs(price - x.Strike))
        contract = contracts[0]
        return contract
    