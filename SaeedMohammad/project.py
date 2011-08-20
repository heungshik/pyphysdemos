#numpy is a numerical array library for Python

import numpy as np

#matplotlib.pyplot is a command style function that allows the user to plot and edit graphs

import matplotlib.pyplot as plt

#math allows the user to use mathematical expressions such as sin, pi, etc...

import math

#Set the x- and y- axes to a certain value. Define dx and dy.

NX = 1000

NY = 1000

dx = 1./NX; dy = 1./NY

#Set the radius of the protruding ridge.

RADIUS = 0.3

# Calculate the number of rows in the ridge.

nrow = int(RADIUS*NY)

print nrow

#Define the width and size of the conducting semicircle.

ridgeWidth = np.zeros(nrow)

#Create a loop for iteration.

for j in range(nrow):
    
    ridgeWidth[j] = int(NX*math.sqrt(RADIUS**2 - ((j+1)*dy)**2 +1e-6)+0.5)
    
print ridgeWidth

#Add a subplot and label each of the sides
fig = plt.figure()

v = fig.add_subplot(111)

v.set_ylabel('Linearly Decreasing Potential')
                
v.set_xlabel('V = 1 Volt')

v.set_title('V = 0 Volts')

v.text(1025,500,'Linearly Decreasing Potential',
        horizontalalignment='left',
        verticalalignment='center',
        rotation = 90)

#Make the potential(v) zero everywhere on the graph

v = np.zeros([NX,NY])

#Set the boundaries of your graph (voltages in this case) to a certain value

#Set the bottom boundary to 1

v[0,:] = 1.

#Set the top boundary to 0

v[-1,:] = 0.

#Set the left and right boundaries to a linearly decreasing potential.

v[:,:] = np.linspace(1.,0.,NY).reshape([NY,1])

#Iterate to solve Laplace equation (by creating a loop). range(n) repeats the loop n times

for istep in range(100000):

#(1:NX-1)means the starting point is 1 unit to the right of the left boundary and the ending point is 1 unit before the right boundary.

#(1:NY-1)means the starting point is 1 unit to above of the bottom boundary and the ending point is 1 unit below the top boundary.

#To find v on every point on the graph, we find the average of the four neighboring values

    v[1:NX-1,1:NY-1] = 0.25*(v[2:NX,1:NY-1] + v[0:NX-2,1:NY-1]

                         + v[1:NX-1,2:NY] +v[1:NX-1,0:NY-2])

    for j in range(nrow):
        
        v[j+1,NX/2-ridgeWidth[j]:NX/2+ridgeWidth[j]] = 1.

#Define the electric field vector components.

ex,ey = np.gradient(-v)

#print is just a tool to check our electric field values.

print (ex,ey)

#Plot the electric field vector using the quiver function. We also enlarge the size of our electric field vectors

plt.quiver(ey,ex,scale=1.)

#contour shows the contour lines.Python's origin is located at the top left corner of the screen; thus, we command Python to move its origin to the bottom left.

plt.contour(v,origin="lower")

#colorbar provides the 'colorbar' on the right

plt.colorbar()

#show simply shows the graph

plt.show()

