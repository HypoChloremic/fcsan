from dash.dependencies import Input, Output
from analyze import Analyze
import dash
import dash_core_components as dcc
import dash_html_components as html
import json


analyze_run = Analyze()
data 		= analyze_run.read()
print(analyze_run.meta)

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure={
            'data': [
                {
                    'x': analyze_run.dataset["FSC-A"].values,
                    'y': analyze_run.dataset["SSC-A"].values,
                    'mode': 'markers',
                    
                },
            ]
        }
    ),

    html.Div([
        dcc.Markdown("""
            **Hover Data**

            Mouse over values in the graph.
        """.replace('   ', '')),
        html.Pre(id='hover-data')
    ]),

    html.Div([
        dcc.Markdown("""
            **Click Data**

            Click on points in the graph.
        """.replace('    ', '')),
        html.Pre(id='click-data'),
    ]),

    html.Div([
        dcc.Markdown("""
            **Selection Data**

            Choose the lasso or rectangle tool in the graph's menu
            bar and then select points in the graph.
        """.replace('    ', '')),
        html.Pre(id='selected-data'),
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-interactions', 'hoverData')])

def display_hover_data(hoverData):
    #
    # This is where you can access the hover data
    # This function will get called automatically when you hover over points
    # hoverData will be equal to an object with that data
    # You can compute something off of this data, and return it to the front-end UI
    # 


    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    [Input('basic-interactions', 'clickData')])
def display_click_data(clickData):
    # Similarly for data when you click on a point
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('selected-data', 'children'),
    [Input('basic-interactions', 'selectedData')])

def display_selected_data(selectedData):
    # Similarly for data when you select a region
    print(selectedData)
    return json.dumps(selectedData, indent=2)

app.run_server(debug=True)