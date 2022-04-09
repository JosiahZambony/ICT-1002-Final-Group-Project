
# Load pandas
import pandas as pd

# Load data
df = pd.read_csv('V-3.csv', index_col=1, parse_dates=True)
df.index = pd.to_datetime(df['Date'])

# Use dash for presenting the data
import dash
import dash_html_components as html
# dash automatically will search for the css file in the assets folder

#Plotly graph
import dash_core_components as dcc
import plotly.express as px

#Declare and create the app
app = dash.Dash(__name__)

# Define the app
# The first html.div will be in charge of the layout and the second will be incharge of the content
app.layout = html.Div(children=[

                      html.Div(className='row',
                       # Sidebar
                               children=[
                                  html.Div(className='four columns div-user-controls',
                                  children = [html.H2('STOCKS PROJECT'),html.P('''Feeling Clueless about stocks? Let us help you out! '''),
                                  html.P('''Enter your stock ticker symbol.''')]),

                                  #HTMl Stuff
                                  html.Div(className='eight columns div-for-charts bg-grey',
                                  children = [

                                  #Main graph to display to users
                                  dcc.Graph(id='timeseries',
                                  config={'displayModeBar': False},
                                  animate=True,
                                  figure=px.line(df,
                                  x='Date',
                                  y='Adj Close',
                                  #color='stock',
                                  template='plotly_dark').update_layout(
                                   {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
                                   )

                                  ]

                                  )])])



# End of the html elements
# Run the app

if __name__ == '__main__':
    app.run_server(debug=True)
