import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

df = pd.read_csv('/content/example_prediction_result.csv')

crypto_symbol = df.columns[1].split('_')[0]
current_time = df.iloc[0, 0]

def predict_future_price(crypto_symbol: str, current_time: int = int(time.time())):
    """
    Given required parameters, predict future price

    :param crypto_symbol: e.g. BTC
    :param current_time: timestamp to start with, usually now
    :return:
    """
    pass
