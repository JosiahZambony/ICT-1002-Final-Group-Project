# This code is used to display the algorithm output to a matplib grpah
# Only run this file after C has successfully run.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import re

import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
import autograd.numpy as npa

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

#def displayGraph():
### Read the updated file

# Three array variable to store the x and y
x1_array = []
x2_array = []
y_array = []

#Iteration array to store iteration number
iteration_array = []

#Interval arrays to store interval values
interval1_array = []
interval2_array = []


with open('result.txt', 'rt') as myfile:
    for myline in myfile:
        x1 = re.search("x = \[(.*), ", myline)
        x2 = re.search(", (.*)\] y = ", myline)
        y = re.search("y = (.*) Interval\[0\]:", myline)
        iteration = re.search("Iteration\[(.*)\]: x =", myline)
        interval1 = re.search("Interval\[0\]: (.*) Interval\[1\]", myline)
        interval2 = re.search("Interval\[1\]: (.*)", myline)
        x1_array.append(float(x1.group(1)))
        x2_array.append(float(x2.group(1)))
        y_array.append(float(y.group(1)))
        iteration_array.append(int(iteration.group(1)))
        interval1_array.append(float(interval1.group(1)))
        interval2_array.append(interval2.group(1))

#plt.plot(iteration_array,interval1_array)
#plt.title('title name')
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.xticks(np.arange(min(iteration_array), max(iteration_array)+1, 1000))
#plt.show()

sx1 = np.array(x1_array, dtype=float)
sx1 = sx1.reshape(-1, 1)
sx2 = np.array(x2_array, dtype=float)
sx2 = sx2.reshape(-1, 1)
sy = np.array(y_array, dtype=float)
sy = sy.reshape(-1, 1)


f  = lambda x_test, y_test: (1.5 - x_test + x_test*y_test)**2 + (2.25 - x_test + x_test*y_test**2)**2 + (2.625 - x_test + x_test*y_test**3)**2
xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2
x_test, y_test = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x_test, y_test)
print(z)
minima = np.array([4., .5])
f(*minima)
minima_ = minima.reshape(-1, 1)
minima_
f(*minima_)


fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)

ax.plot(y_array, linestyle='--', marker='o', color='orange')
ax.plot(len(y_array)-1, y_array[-1], 'ro')
ax.set(title='Y Value During Optimization Process', xlabel='Iterations', ylabel='Y')

ax = fig.add_subplot(elev=50, azim=-50, projection='3d')
ax.plot_surface(x_test, y_test, z, norm=LogNorm(), rstride=1, cstride=1,
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
ax.plot(x1_array,x2_array,y_array, color='red')


plt.show()