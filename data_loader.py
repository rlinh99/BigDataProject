import pandas as pd
import numpy as np


def load_data(type):
    if type == 'train':
        path = "data/training.csv"
    return pd.read_csv(type)
train_dataset = pd.read_csv("data/training.csv")
test_dataset = pd.read_csv("data/testing.csv")
validation_dataset = pd.read_csv("data/validation.csv")