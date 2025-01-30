import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def predict_future_price(crypto_symbol: str, current_time: int = int(time.time())):
    """
    Given required parameters, predict future price

    :param crypto_symbol: e.g. BTC
    :param current_time: timestamp to start with, usually now
    :return:
    """
    pass