import pandas as pd
import numpy as np


def load_data(type):
    path = 'data/'
    if type == 'train':
        path = path+'training.csv'
    if type == 'validation':
        path = path + 'validation.csv'
    return pd.read_csv(path)


