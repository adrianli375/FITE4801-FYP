from AlgorithmImports import *
from QuantConnect.DataSource import *

import pandas as pd
import numpy as np
from datetime import timedelta

from loadModel import load_model


# The trading strategy based on cryptocurrency metadata with machine learning. 
# NOTE: This algorithm is not published in the report and final results, 
# and it is only involved in the preliminary development process. 
class BitcoinMetadataAlgorithm(QCAlgorithm):
    
    def Initialize(self) -> None:
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. 
        All algorithms must be initialized before performing testing.'''
        self.SetStartDate(2019, 9, 1)   # Set Start Date
        self.SetEndDate(2022, 12, 31)    # Set End Date
        self.SetCash(1000000)
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.resolution = Resolution.Daily

        ### Extra params to include
        self.tradePeriod = 3
        self.traded = False
        self.placedTrade = self.Time
        self.tradeConfidenceLevel = 0.6
        self.tradeHistory = 252 * 5
        self.npercent = 0.025
        self.stopLossPercent = 0.1
        self.sellatnpercent = True
        self.sellatndays = True
        self.extendWhenSignalled = True
        self.updateExePriceWhenExtend = True # self.extendWhenSignalled must be true
        self.sellWhenSignalChange = True

        ### ML params
        ### get the model name: choose one from {'rf', 'lr', 'svm', 'gb', 'knn', 'stacking', 'votinghard', 'votingsoft'}
        self.modelName = 'votinghard'
        self.predictionMechanism = 'soft' # one from {'soft', 'hard'}, 'soft' is preferred
        
        ### this crypto data is only available from the Bitfinex market
        self.ticker = self.AddCrypto("BTCUSD", self.resolution, Market.Bitfinex).Symbol
        ### Requesting data
        self.metadataSymbol = self.AddData(BitcoinMetadata, self.ticker).Symbol 
        
        ### other fixed params
        self.trainRatio = 0.8
        self.model = None
        self.lastDemandSupply = None
        self.exePrice = 0

        self.trainModel()

    def trainModel(self) -> None:
        '''Trains the model. '''
        ### preprocessing
        ### Historical data
        metadata = self.History(BitcoinMetadata, self.metadataSymbol, self.tradeHistory, self.resolution)
        history = self.History(self.ticker, self.tradeHistory, self.resolution)#[['close', 'volume']]
        self.Debug(history.shape)
        self.Debug(f"We got {len(history)} items from our history request for {self.ticker} Blockchain Bitcoin Metadata")
        # self.Debug(f"The time is {self.Time}")

        # obtain the future close price and the returns
        history['closeAfterFewDays'] = history['close'].shift(-self.tradePeriod)
        history['returns'] = history['closeAfterFewDays'] / history['close'] - 1
        history.dropna(inplace=True)

        # obtain the signal (buy, sell or hold)
        history['signal'] = history['returns'].apply(lambda x: 1 if x > self.npercent else (-1 if x < -self.npercent else 0))
        
        # update the metadata dataframe
        metadata['signal'] = np.nan
        metadataSignalIdx = list(metadata.columns).index('signal')
        historySignalIdx = list(history.columns).index('signal')
        for i in range(len(history)):
            metadata.iat[i, metadataSignalIdx] = history.iat[i, historySignalIdx]
        metadata.dropna(inplace=True)
        
        ### modelling
        trainSize = int(self.tradeHistory * self.trainRatio)
        X_train = metadata.drop(columns=['signal'])[:trainSize].values
        y_train = metadata['signal'][:trainSize]

        X_valid = metadata.drop(columns=['signal'])[trainSize:].values
        y_valid = metadata['signal'][trainSize:]

        self.model = load_model(self.modelName)
        self.model.fit(X_train, y_train)

        self.Log(f'The validation score of the model is {self.model.score(X_valid, y_valid)}')
    

    def OnData(self, slice: Slice) -> None:
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            slice: Slice object keyed by symbol containing the stock data
        '''

        ### Retrieving data (Bitcoin metadata)
        data = slice.Get(BitcoinMetadata)
        if not data.ContainsKey(self.metadataSymbol):
            return
        latestMetadata = data[self.metadataSymbol]

        # obtain the latest series of Bitcoin metadata (total 23 features)
        X_pred = np.array([
            latestMetadata.AverageBlockSize, latestMetadata.BlockchainSize,
            latestMetadata.CostPercentofTransactionVolume, latestMetadata.CostPerTransaction, 
            latestMetadata.Difficulty, 
            latestMetadata.EstimatedTransactionVolume, latestMetadata.EstimatedTransactionVolumeUSD, 
            latestMetadata.HashRate, latestMetadata.MarketCapitalization, 
            latestMetadata.MedianTransactionConfirmationTime, latestMetadata.MinersRevenue, 
            latestMetadata.MyWalletNumberofTransactionPerDay, latestMetadata.MyWalletNumberofUsers, 
            latestMetadata.MyWalletTransactionVolume, latestMetadata.NumberofTransactionperBlock, 
            latestMetadata.NumberofTransactions, latestMetadata.NumberofTransactionsExcludingPopularAddresses, 
            latestMetadata.NumberofUniqueBitcoinAddressesUsed, latestMetadata.TotalBitcoins, 
            latestMetadata.TotalNumberofTransactions, latestMetadata.TotalOutputVolume, 
            latestMetadata.TotalTransactionFees, latestMetadata.TotalTransactionFeesUSD
        ])
        
        if self.metadataSymbol in data and data[self.metadataSymbol] != None:
            # get market data
            currentPrice = slice.Bars[self.ticker].Close
            currentDemandSupply = data[self.metadataSymbol].NumberofTransactions / data[self.metadataSymbol].HashRate
            avgPrice = self.Portfolio[self.ticker].AveragePrice
            positions = self.Portfolio[self.ticker].Quantity
            cashAvailable = self.Portfolio.Cash

            # get signals
            predictedSignal = self.model.predict(np.reshape(X_pred, (1, -1))).astype(int)[0]
            positiveSignal = predictedSignal == 1
            negativeSignal = predictedSignal == -1
            if self.predictionMechanism == 'hard':
                modelConfidence = predictedSignal
            elif self.predictionMechanism == 'soft': # 'soft'
                try:
                    modelConfidenceArr = self.model.predict_proba(np.reshape(X_pred, (1, -1)))
                    if positiveSignal:
                        modelConfidence = modelConfidenceArr[0][-1]
                    elif negativeSignal:
                        modelConfidence = modelConfidenceArr[0][0]
                    else: # neutral signal
                        modelConfidence = 0
                except AttributeError: # if predict_proba is not available
                    modelConfidence = predictedSignal

            # comparing the average transaction-to-hash-rate ratio changes, we will buy bitcoin or hold cash
            #if self.lastDemandSupply != None and currentDemandSupply > self.lastDemandSupply:
            if abs(positions * currentPrice) > 100:
                # update exePrice
                if self.exePrice == 0:
                    self.exePrice = avgPrice
                sell = False

                # if sell at n days is true, we sell it
                if self.sellatndays:
                    if (self.placedTrade + timedelta(days=self.tradePeriod)) >= self.Time:
                        sell = True
                
                # check if the signal is extended
                if self.extendWhenSignalled:
                    if (modelConfidence >= self.tradeConfidenceLevel and positions > 0) or (modelConfidence <= -self.tradeConfidenceLevel and positions < 0):
                        self.placedTrade = self.Time
                        if self.updateExePriceWhenExtend:
                            self.exePrice = currentPrice
                        self.Log('time extended')

                # check if signal is changed to sell
                if self.sellWhenSignalChange:
                    if (modelConfidence >= self.tradeConfidenceLevel and positions > 0) or (modelConfidence <= -self.tradeConfidenceLevel and positions < 0):
                        sell = True
                        self.Log('signal changed to sell')

                # take profit
                if not sell and self.sellatnpercent:
                    self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
                if positions > 0:
                    self.LimitOrder(self.ticker, -positions, self.exePrice * (1 + self.npercent), tag='take profit from long position')
                else:
                    self.LimitOrder(self.ticker, -positions, self.exePrice * (1 - self.npercent), tag='take profit from short position')
                
                # stop loss
                if not sell and self.stopLossPercent > 0:
                    self.DefaultOrderProperties.TimeInForce = TimeInForce.Day
                    if positions > 0:
                        if currentPrice < self.exePrice * (1 - self.stopLossPercent):
                            self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                            self.MarketOrder(self.ticker, -positions)
                            self.Log('stop loss triggered from long position')
                        else:
                            self.StopLimitOrder(self.ticker, -positions, self.exePrice * (1 - self.stopLossPercent), self.exePrice * (1 - self.stopLossPercent))
                    else:
                        if currentPrice > self.exePrice * (1 + self.stopLossPercent):
                            self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                            self.MarketOrder(self.ticker, -positions)
                            self.Log('stop loss triggered from short position')
                        else:
                            self.StopLimitOrder(self.ticker, -positions, self.exePrice * (1 + self.stopLossPercent), self.exePrice * (1 + self.stopLossPercent))
                
                # if sell triggered
                if sell:
                    self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                    self.MarketOrder(self.ticker, -positions)
            else:
                self.exePrice = 0
                self.DefaultOrderProperties.TimeInForce = TimeInForce.GoodTilCanceled
                # place market orders when the model is confident enough
                if modelConfidence >= self.tradeConfidenceLevel:
                    quantity = int(cashAvailable / currentPrice)
                    self.MarketOrder(self.ticker, quantity)
                    self.placedTrade = self.Time
                elif modelConfidence <= -self.tradeConfidenceLevel:
                    quantity = int(cashAvailable / currentPrice)
                    quantity = int(cashAvailable / currentPrice)
                    self.MarketOrder(self.ticker, -quantity)
                    self.placedTrade = self.Time
                            
            # finally, update the latest demand supply of the blockchain
            self.lastDemandSupply = currentDemandSupply
