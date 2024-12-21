import datetime

import keras.models
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
from sklearn import preprocessing
import random
from collections import deque
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Bidirectional, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import joblib

seq = 192

df = pd.read_pickle("../data/preprocessed3/load.pkl")
df["time"] = df.index.hour
df["month"] = df.index.month
#df2 = df[["S_TOT", "date_month_sin", "date_month_cos" ,"date_is_holiday", "date_is_weekday", "date_is_weekend", "date_season", "TARGET"]]
df2 = df[["S_TOT","time","month", "TARGET"]]
#df2 = df[["S_TOT", "TARGET"]]
X_train = df2[:int(len(df2) * 0.8)]
X_test = df2[int(len(df2) * 0.8):]
min_max_scaler = joblib.load("scaler_target.save")
model = keras.models.load_model("mymodel")
print(X_test)
c = 0
set_prev = False
for ind, row in X_test.iterrows():
    dee = pd.DataFrame(columns=['preds'])
    if c > seq:
        if ind.hour == 0 and ind.minute == 0:
            rel_df = X_test.iloc[c-seq:c, 0:3]
            set_prev = True
        if set_prev:
            d = rel_df.to_numpy().reshape((1,192,3))
            pred = model.predict(d, verbose=0)
            new_ind = ind + datetime.timedelta(minutes=15)
            df2 = pd.DataFrame({"S_TOT": pred[0][0], "time": X_test.iloc[c+1, 1], "month": X_test.iloc[c+1, 2]}, index=[new_ind])
            rel_df = rel_df.append(df2, ignore_index=True)
            rel_df = rel_df.iloc[1:, :]
            X_test.loc[ind]["preds"] = min_max_scaler.inverse_transform(pred)[0][0]

    c +=1

print(X_test)