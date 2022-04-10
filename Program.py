import os
import sys
import dash
import dash_html_components as html

import dash_core_components as dcc
import plotly.express as px
app = dash.Dash(__name__)


# use os.system to compile c scripts and run c script
os.system('gcc algorithm.c -o execute');
string = './execute '
os.system(string);
