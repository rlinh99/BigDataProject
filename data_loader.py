import pandas as pd
import numpy as np
# from keras.preprocessing.text import Tokenizer


def load_data(type):
    path = 'data/'
    if type == 'train':
        path = path + 'training.csv'
    if type == 'validation':
        path = path + 'validation.csv'
    return pd.read_csv(path)


train_data = load_data('train')
test_data = load_data('validation')