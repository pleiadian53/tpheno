# encoding: utf-8

# Learn about API authentication here: https://plot.ly/pandas/getting-started
# Find your api_key here: https://plot.ly/settings/api

import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np # for generating random data


print('> basic histogram')
N = 500
x = np.linspace(0, 1, N)
y = np.random.randn(N)
df = pd.DataFrame({'x': x, 'y': y})
df.head()

data = [go.Histogram(y=df['y'])]

# IPython notebook
# py.iplot(data, filename='pandas/simple-histogram')

url = py.plot(data, filename='simple-histogram')