import numpy as np
import pandas as pd

# Plots
# ==============================================================================

from joblib import dump, load
import plotly.express as px


data_test = pd.read_csv("test.csv", index_col=0)


print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

forecaster = load('models/xgboost_model.py')
seq_len = 192
c = 0
d = data_test
de = pd.DataFrame(columns=['preds'])
print(len(data_test))
for ind, row in data_test.iterrows():
    dee = pd.DataFrame(columns=['preds'])
    if c > seq_len:
        if ind.hour == 0 and ind.minute == 0:
            dee["preds"] = forecaster.predict(steps=96, last_window=d.iloc[c-seq_len:c,0])
            dee["datetime"] = d.iloc[c:c+96].index
            de = de.append(dee, ignore_index=True)
    c +=1
de = de.set_index("datetime")
d["preds"] = de["preds"]
d.dropna(inplace=True)
d["preds"] = d["preds"].astype(float)


fig = px.line(d, x=d.index,  y=["TARGET", "preds"] )
fig.show()