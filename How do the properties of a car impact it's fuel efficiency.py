import pandas
from pprint import pprint as pp
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import sklearn
from sklearn.linear_model import LinearRegression

#Import data; it is in .data format and is white space delimited. The dataset does not come with a header so we must
# manually add them. This information can be found in the auto-mpg.names file
names = ['mpg', 'cylinders', 'displacement', 'horse power', 'weight', 'acceleration', 'model year', 'origin', 'car name']
mpg = pandas.read_table("auto-mpg.data", delim_whitespace=True, names=names)
pp(mpg.head())

#pandas.to_numeric(mpg['horse power'], downcast='float')
#let's look at the different car attributes in relation to mileage
pp(mpg.info())
# use plotly to look at the info and how each series in the data frame impact mpg
data = []
for n in names:
    t = 'trace' + n
    trace = go.Scatter(
        x = mpg['mpg'],
        y = mpg[n],
        name = n + ' vs mpg',
        mode = 'line+markers',
        hoverinfo = 'y'
    )
    t = trace
    data.append(t)
layout = dict(title = "Dataframes Vs MPG", xaxis=dict(
        #label = 'MPG',
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        autotick=False,
        ticks='outside',
        tickcolor='rgb(204, 204, 204)',
        tickwidth=2,
        ticklen=5,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',)))

#fig = go.Figure(data=data, layout = layout)
#py.plot(fig,layout=layout)
#Use SKlearn to fit the most linearly correlated series to predict mpg
#linear regressyion assumes a relationship of prediction = slope*attribute + y_intercept
lr = LinearRegression()
lr.fit(mpg[['weight']].values, mpg['mpg'].values)
weightLr = lr.predict(mpg[['weight']])

trace1 = go.Scatter(
        x = mpg['mpg'],
        y = mpg['weight'],
        name = n + ' vs mpg',
        mode = 'line+markers',
        hoverinfo = 'y'
    )
tracepred = go.Scatter(
        x = weightLr,
        y = mpg['weight'],
        name = n + ' vs mpg',
        mode = 'line+markers',
        hoverinfo = 'y'
    )
data = [trace1,tracepred]
fig = go.Figure(data=data, layout = layout)
py.plot(fig,layout=layout)
mse = sklearn.metrics.mean_squared_error(mpg['mpg'],weightLr)
