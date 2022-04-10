# This code is used to display the algorithm output to a matplib grpah
# Only run this file after C has successfully run.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


#def displayGraph():
### Read the updated file

# Two array variable to store the x and y
x_array = []
y_array = []

with open('result2.txt', 'rt') as myfile:
    for myline in myfile:
        x = myline[23:41]
        y = myline[46:55]
        x_array.append(x)
        y_array.append(y)
