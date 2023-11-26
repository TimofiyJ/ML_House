import pickle
import numpy as np
from settings.constants import SAVED_ESTIMATOR


class Predictor:
    def __init__(self):
        self.loaded_estimator = pickle.load(open(SAVED_ESTIMATOR, 'rb'))

    def predict(self, data):
        return self.loaded_estimator.predict(data)
    def mean_absolute_percentage_error(val_y, api_predict):
        val_y, api_predict = np.array(val_y), np.array(api_predict)
        return np.mean(np.abs((val_y - api_predict) / val_y)) * 100