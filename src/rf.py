import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5


# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

from joblib import dump, load

# Warnings configuration
# ==============================================================================
import warnings
# warnings.filterwarnings('ignore')

df = pd.read_pickle("E:/bda-analytics-challenge/data/not_scaled/load.pkl")
df["time"] = df.index.hour
df["month"] = df.index.month
#df2 = df[["S_TOT", "date_month_sin", "date_month_cos" ,"date_is_holiday", "date_is_weekday", "date_is_weekend", "date_season", "TARGET"]]
df2 = df[["S_TOT","time","month", "TARGET"]]
data = df2
split = 0.95
data_train = data[:int(len(data) * split)]
data_test = data[int(len(data) * split):]

print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

fig, ax=plt.subplots(figsize=(9, 4))
data_train['S_TOT'].plot(ax=ax, label='train')
data_test['S_TOT'].plot(ax=ax, label='test')
ax.legend()


steps = 5000
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # This value will be replaced in the grid search
             )

# Lags used as predictors
lags_grid = [32, 192]

# Regressor's hyperparameters
param_grid = {'n_estimators': [500]
              }

results_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = data_train['TARGET'],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        refit              = True,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(data_train)*0.5),
                        fixed_train_size   = False,
                        return_best        = True,
                        verbose            = False
               )

print(results_grid)





