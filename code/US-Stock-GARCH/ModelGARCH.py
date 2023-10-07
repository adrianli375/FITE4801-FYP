from AlgorithmImports import *
from arch import arch_model


class GarchModel:
    def __init__(self, y: np.array, last_obs_value: float, p: int=1, q: int=1):
        self.y_train = np.diff(np.log(y))
        self.last_obs_value = last_obs_value
        self.p = p
        self.q = q
    
    def fit(self, **kwargs):
        model = arch_model(self.y_train, rescale=False, p=self.p, q=self.q, **kwargs)
        self.model = model.fit(show_warning=False)
    
    def get_model_summary(self):
        print(self.model.summary())
    
    def predict_forecasts(self, n_steps=1, print_output=True):
        forecast = self.model.forecast(horizon=n_steps)
        forecast_log_return = forecast.mean.iloc[-1, 0]
        # for stability issues
        forecast_value = self.last_obs_value * np.exp(forecast_log_return)
        forecast_std = np.sqrt(forecast.variance.iloc[-1, 0])
        if print_output:
            print(f'[{datetime.now()}] Predicted value: {forecast_value}')
        return forecast_value, forecast_std
