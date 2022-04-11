# This code is used to display the algorithm output to a matplib grpah
# Only run this file after C has successfully run.
import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from matplotlib.colors import LogNorm


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
domain_min_array = []
domain_max_array = []
global_1_array = []
global_2_array = []


with open('result.txt', 'rt') as myfile:
    for myline in myfile:
        x1 = re.search("x = \[X1\((.*)\), X2", myline)
        x2 = re.search(", X2\((.*)\)\] y = ", myline)
        y = re.search("y\((.*)\)y", myline)
        domain_min = re.search("DR1\[\((.*)\)\]DR1, DR2", myline)
        domain_max = re.search("DR2\[\((.*)\)\]DR2", myline)
        global_1 = re.search("GM1\[\((.*)\)\]GM1", myline)
        global_2 = re.search("GM2\[\((.*)\)\]GM2", myline)
        iteration = re.search("Iteration\[(.*)\]: x =", myline)
        x1_array.append(float(x1.group(1)))
        x2_array.append(float(x2.group(1)))
        domain_min_array.append(float(domain_min.group(1)))
        domain_max_array.append(float(domain_max.group(1)))
        global_1_array.append(float(global_1.group(1)))
        global_2_array.append(float(global_2.group(1)))
        y_array.append(float(y.group(1)))
        iteration_array.append(int(iteration.group(1)))

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
xmin, xmax, xstep = domain_min_array[1], domain_max_array[1], .2
ymin, ymax, ystep = domain_min_array[1], domain_max_array[1], .2
x_test, y_test = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x_test, y_test)
minima = np.array([global_1_array[1], global_2_array[1]])
f(*minima)
minima_ = minima.reshape(-1, 1)
minima_
f(*minima_)

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)

ax.plot(y_array, linestyle='--', marker='o', color='orange')
ax.plot(len(y_array)-1, y_array[-1], 'ro')
ax.set(title='Reduction of Y Value During Optimization Process', xlabel='Iterations', ylabel='Y')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(elev=50, azim=-50, projection='3d')
ax.plot_surface(x_test, y_test, z, norm=LogNorm(), rstride=1, cstride=1,
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
ax.plot(*minima_,f(*minima_),'r*', color='red')


plt.show()