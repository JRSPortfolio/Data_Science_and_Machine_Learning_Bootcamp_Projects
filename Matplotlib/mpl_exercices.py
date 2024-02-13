'''
Follow the instructions to recreate the plots using this data:
import numpy as np
x = np.arange(0,100)
y = x*2
z = x**2
'''

import numpy as np

x = np.arange(0,100)
y = x*2
z = x**2

# Import matplotlib.pyplot as plt and set %matplotlib inline if you are using the jupyter notebook.
import matplotlib.pyplot as plt

## Exercise 1
# Follow along with these steps: 
# Create a figure object called fig using plt.figure()
# Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax. 
# Plot (x,y) on that axes and set the labels and titles to match the plot below:
ex1_fig = plt.figure()
ax = ex1_fig.add_axes([0.1, 0.1, 0.85, 0.85]) #at [0,0,1,1] legends don't fit the canvas 
ax.set_title('Exercise 1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.plot(x, y, 'b')


## Exercise 2
# Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2] respectively.
# plot (x,y) on both axes. And call your figure object to show it.
fig = plt.figure(figsize = (10, 7))
ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
ax2 = fig.add_axes([0.2, 0.5, 0.2, 0.2])

ax1.set_title('Exercise 2 - XY')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.set_title('Exercise 2 - XY')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

ax1.plot(x, y, 'b', lw = 0.5)
ax2.plot(x, y, 'c', lw = 0.5)

## Exercise 3
# Create the plot below by adding two axes to a figure object at [0,0,1,1] and [0.2,0.5,.4,.4]
# Now use x,y, and z arrays to recreate the plot below. Notice the xlimits and y limits on the inserted plot:
fig = plt.figure(figsize = (10, 7))
ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
ax2 = fig.add_axes([0.2, 0.5, 0.4, 0.4])

ax1.set_title('Exercise 3 - XZ')
ax1.set_xlabel('X')
ax1.set_ylabel('Z')

ax2.set_title('Exercise 3 - ')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

ax2.set_xlim([20, 22])
ax2.set_ylim([30, 50])

ax1.plot(x, z, 'm', lw = 0.5)
ax2.plot(x, y, 'g', lw = 0.5)

## Exercise 4
# Use plt.subplots(nrows=1, ncols=2) to create the plot below.
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 7))

# Now plot (x,y) and (x,z) on the axes. Play around with the linewidth and style
axes[0].plot(x, y, 'b', lw = 3, ls = '--')
axes[1].plot(x, z, 'r', lw = 3)

#See if you can resize the plot by adding the figsize() argument in plt.subplots() are copying and pasting your previous code
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 2))

axes[0].plot(x, y, 'b', lw = 3, ls = '--')
axes[1].plot(x, z, 'r', lw = 3)


plt.show()