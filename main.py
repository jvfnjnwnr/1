from yahoo_fin.stock_info import get_data
from sklearn import linear_model
import pandas as pd
import numpy
data = get_data("uvxy")
time = list(range(1, len(data['open'])+1))
data['open_lagged'] = data.groupby(['ticker'])['open'].shift(1)
data['high_lagged'] = data.groupby(['ticker'])['high'].shift(1)
data['low_lagged'] = data.groupby(['ticker'])['low'].shift(1)
data['close_lagged'] = data.groupby(['ticker'])['close'].shift(1)
reg = linear_model.BayesianRidge()
X = pd.DataFrame(data = [time, data['open_lagged'], data['high_lagged'], data['low_lagged'], data['close_lagged']])
X = numpy.transpose(X.drop(X.columns[[0]], axis=1))
open = data['open'].iloc[1:2332]
reg = reg.fit(X, open)
print(reg.score(X, open))
