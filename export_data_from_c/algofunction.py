# This code is used to display the algorithm output to a matplib grpah
# Only run this file after C has successfully run.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import re


#def displayGraph():
### Read the updated file

# Two array variable to store the x and y
x_array = []
y_array = []

#Iteration array to store iteration number
iteration_array = []

#Interval arrays to store interval values
interval1_array = []
interval2_array = []


with open('result2.txt', 'rt') as myfile:
    for myline in myfile:
        x = re.search("x = \[(.*)\] y =", myline)
        y = re.search("y = (.*) Interval\[0\]", myline)
        iteration = re.search("Iteration\[(.*)\]: x =", myline)
        interval1 = re.search("Interval\[0\]: (.*) Interval\[1\]", myline)
        interval2 = re.search("Interval\[1\]: (.*)", myline)
        x_array.append(x.group(1))
        y_array.append(y.group(1))
        iteration_array.append(iteration.group(1))
        interval1_array.append(interval1.group(1))
        interval2_array.append(interval2.group(1))

print(interval1_array)