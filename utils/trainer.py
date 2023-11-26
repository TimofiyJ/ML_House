from sklearn.ensemble import RandomForestRegressor
import numpy as np

class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return RandomForestRegressor().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
    def mean_absolute_percentage_error(val_y, api_predict):
        val_y, api_predict = np.array(val_y), np.array(api_predict)
        return np.mean(np.abs((val_y - api_predict) / val_y)) * 100

