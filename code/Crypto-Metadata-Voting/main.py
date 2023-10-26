from AlgorithmImports import *
from QuantConnect.DataSource import *

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

class BitcoinMachineLearningAlgorithm(QCAlgorithm):
    
    def Initialize(self) -> None:
        self.SetStartDate(2019, 9, 1)   # Set Start Date
        self.SetEndDate(2022, 12, 31)    # Set End Date
        self.SetCash(1000000)
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.resolution = Resolution.Daily

        ### Extra params to include
        self.trainHistory = 180
        self.trainRatio = 0.75
        self.retrainPeriod = 30
        self.tradePeriod = 1
        self.returnPercentage = 0.03
        self.sellPositionsRatio = 0.75
        self.takeProfitLevel = 1.5
        self.stopLossLevel = 0.9
        
        ### this crypto data is only available from the Bitfinex market
        self.ticker = self.AddCrypto("BTCUSD", self.resolution, Market.Bitfinex).Symbol
        ### Requesting data
        self.metadataSymbol = self.AddData(BitcoinMetadata, self.ticker).Symbol 
        
        ### other fixed params
        self.model = None
        self.trainedCounter = 0
        self.lastDemandSupply = None

        self.trainModel()

    def trainModel(self) -> None:
        ### preprocessing
        ### Historical data
        metadata = self.History(BitcoinMetadata, self.metadataSymbol, self.trainHistory, self.resolution)
        history = self.History(self.ticker, self.trainHistory, self.resolution)#[['close', 'volume']]
        self.Debug(history.shape)
        self.Debug(f"We got {len(history)} items from our history request for {self.ticker} Blockchain Bitcoin Metadata")
        # self.Debug(f"The time is {self.Time}")
        history['closeAfterFewDays'] = history['close'].shift(-self.tradePeriod)
        history['returns'] = history['closeAfterFewDays'] / history['close'] - 1
        history.dropna(inplace=True)
        history['signal'] = history['returns'].apply(lambda x: 1 if x > self.returnPercentage else (-1 if x < -self.returnPercentage else 0))
        
        metadata['signal'] = np.nan
        metadataSignalIdx = list(metadata.columns).index('signal')
        historySignalIdx = list(history.columns).index('signal')
        for i in range(len(history)):
            metadata.iat[i, metadataSignalIdx] = history.iat[i, historySignalIdx]
        metadata.dropna(inplace=True)
        
        ### modelling
        trainSize = int(self.trainHistory * self.trainRatio)
        X_train = metadata.drop(columns=['signal'])[:trainSize].values
        y_train = metadata['signal'][:trainSize]

        X_valid = metadata.drop(columns=['signal'])[trainSize:].values
        y_valid = metadata['signal'][trainSize:]

        self.model = make_pipeline(
            StandardScaler(), 
            VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=3327)),
                    ('lr', LogisticRegression(random_state=3327)),
                    ('svm', SVC(random_state=3327))
                ]
            )
        )
        self.model.fit(X_train, y_train)

        self.Debug(f'The validation score of the model is {self.model.score(X_valid, y_valid)}')
        self.trainedCounter = 0
    
    def OnData(self, slice: Slice) -> None:
        # increment the train counter
        self.trainedCounter += 1
        if self.trainedCounter >= self.retrainPeriod:
            self.trainModel()
        ### Retrieving data
        data = slice.Get(BitcoinMetadata)
        if not data.ContainsKey(self.metadataSymbol):
            return
        latestMetadata = data[self.metadataSymbol]
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

            # get signals
            predictedSignal = self.model.predict(np.reshape(X_pred, (1, -1)))[0]
            positiveSignal = predictedSignal == 1
            negativeSignal = predictedSignal == -1
            if avgPrice > 0:
                takeProfit = currentPrice > avgPrice * self.takeProfitLevel
                stopLoss = currentPrice < avgPrice * self.stopLossLevel
            else:
                takeProfit, stopLoss = False, False

            # comparing the average transaction-to-hash-rate ratio changes, we will buy bitcoin or hold cash
            if self.lastDemandSupply != None and currentDemandSupply > self.lastDemandSupply and positiveSignal:
                buyQuantity = int(self.Portfolio.Cash / currentPrice)
                if positions <= 2:
                    self.MarketOrder(self.ticker, buyQuantity)
                elif takeProfit:
                    self.MarketOrder(self.ticker, -int(positions * self.sellPositionsRatio))
                else:
                    self.MarketOrder(self.ticker, buyQuantity)
            elif stopLoss:
                self.Liquidate()
            elif negativeSignal or takeProfit:
                self.MarketOrder(self.ticker, -int(positions * self.sellPositionsRatio))
            self.lastDemandSupply = currentDemandSupply
