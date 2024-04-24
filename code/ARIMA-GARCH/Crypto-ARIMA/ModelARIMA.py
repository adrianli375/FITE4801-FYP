from AlgorithmImports import *
from datetime import datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# ARIMA model class
class ArimaModel:
    def __init__(self, y_train: np.array, order: tuple, percent_ci: float):
        '''Initializer method. 
        Arguments:
            y_train: A numpy array containing the time-series data. 
            order: The ARIMA model order. 
            percent_ci: The percentage to calculate the confidence interval of the predicted quantities. 
        '''
        self.y_train = y_train
        self.order = order
        self.percent_ci = percent_ci
    
    def fit(self, **kwargs):
        '''Fits the ARIMA model based on the data. '''
        arima = ARIMA(self.y_train, order=self.order, **kwargs)
        self.model = arima.fit()
    
    def get_model_summary(self):
        '''Obtains the model summary results. '''
        print(self.model.summary())
    
    def predict_forecasts(self, n_steps=1, print_output=True):
        '''Predicts the forecasts based on the fitted model. 
        Arguments:
            n_steps: The number of steps ahead to forecast. Default: 1. 
            print_output: Whether to print the output or not. Default: True. 
        
        Returns: The predicted point estimate and the upper & lower bound of the confidence interval estimates. 
        '''
        forecast = self.model.get_forecast(steps=n_steps)
        forecast_point_est = round(forecast.predicted_mean[0], 2)
        forecast_prediction_int = forecast.conf_int(alpha=1-self.percent_ci)[0]
        lower_est = round(forecast_prediction_int[0], 2)
        upper_est = round(forecast_prediction_int[1], 2)
        if print_output:
            print(f'[{datetime.now()}] Predicted value: {forecast_point_est}')
            print(f'[{datetime.now()}] Predicted {self.percent_ci * 100}% confidence interval: '
                  f'[{lower_est}, {upper_est}]')
        return lower_est, forecast_point_est, upper_est
