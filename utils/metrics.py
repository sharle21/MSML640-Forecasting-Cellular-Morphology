from sklearn.metrics import mean_squared_error
import numpy as np

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def cosine_sim(y_true, y_pred):
    return np.mean(np.sum(y_true * y_pred, axis=1) /
                   (np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)))
