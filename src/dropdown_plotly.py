import plotly
from plotly.graph_objs import Scattergl, Layout
import plotly.figure_factory as ff
import plotly.graph_objs as go
import pandas as pd

plotly.offline.init_notebook_mode(connected=True)

columns = ["x","y","z"]
x = [1,2,3,4,5,6,7,8,9,10]
y = [2,3,4,5,6,7,8,9,10,11]
z = [3,4,5,6,7,8,9,10,11,12]

d    = [x,y,z]
data = []
for i in d:
  for k in d: 
      data.append(go.Scatter(x=i, y=k, name=f"{i}vs{k}"))

colors = ['#ffaeb9', '#ffb6c0', '#ffbec7', '#ffc6ce', '#ffced5',
          '#ffd6dc', '#ffdee3', '#ffe6ea', '#ffeef1', '#fff6f8']
color_buttons = []
column_buttons_x = []
column_buttons_y = []

for i in colors:
    color_buttons.append(
        dict(args=['line.color', i],
             label=i, method='restyle')
    )
for j in columns:
    column_buttons_x.append(
        dict(args=['x',j],
            label=j,method='update')
    )
for k in columns:
    column_buttons_y.append(
        dict(args=['y',k],
            label=k,method='update')
    )

layout = Layout(

    annotations=[dict(text='Change Color',
                      x=-0.25, y=0.83,
                      xref='paper', yref='paper',
                      showarrow=False)],

    updatemenus=list([
        dict(x=-0.1, y=0.7,
             yanchor='middle',
             bgcolor='c7c7c7',
             buttons=list(color_buttons)),
        dict(x=-0.1,y=0.5,
            yanchor = 'middle',
            bgcolor = 'c7c7c7',
            buttons=list(column_buttons_x)),
        dict(x=-0.1,y=0.3,
            yanchor = 'middle',
            bgcolor = 'c7c7c7',
            buttons=list(column_buttons_y))
    ])
)

trace = go.Scatter(
    x=[j],
    y=[k], 
    mode='markers'
)

fig  = dict(data=data, layout=layout)
plotly.offline.plot(fig)