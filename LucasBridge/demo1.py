# -*- coding: utf-8 -*-
import EMDynamics
import numpy as np
import matplotlib.pyplot as plt
import time
import EMVisualization

if __name__ == '__main__':
    print "EM example “demo 1”"
    

    edgeLength = 0.01
    timeStep = 0.3 * edgeLength / EMDynamics.speedOfLight
    N = 20
    extent = edgeLength * N * 0.5
    
    fields = EMDynamics.EMFields((N,N,N), edgeLength)
    advancer = EMDynamics.WeaveCDynamics(fields)
#    advancer = EMDynamics.PurePythonDynamics(fields)
    
    viz = EMVisualization.DirectVisualization(fields)
    
    emax = 1.0
    fields.efield[10,8:13,10,1] = 10*emax
    
    scale = 50
    
    time = 0.0
    
    for i in range(100):
        
        plt.clf()
        
        viz.calculate()
        plt.quiver(viz.EXX*1e2, viz.EXY*1e2, viz.EX, viz.zeros, 
                   width=0.002, scale=scale, color="blue", pivot="middle")
        plt.quiver(viz.EYX*1e2, viz.EYY*1e2, viz.zeros, viz.EY, 
                   width=0.002, scale=scale, color="blue", pivot="middle")

        sizeo = 10.0 / emax * np.where(viz.BZ>0, viz.BZ, 0.0).flatten()
        sizex = 10.0 / emax * np.where(viz.BZ<0, -viz.BZ, 0.0).flatten()
        plt.scatter(viz.BZY.repeat(N, 0).flatten()*1e2, 
                    viz.BZX.repeat(N, 1).flatten()*1e2,
                    s = sizeo, marker="o", edgecolors="red", facecolors="none")
        plt.scatter(viz.BZY.repeat(N, 0).flatten()*1e2, 
                    viz.BZX.repeat(N, 1).flatten()*1e2,
                    s = sizex, marker="x", c="red")
    
        plt.axis(xmin=-extent*1e2, ymin=-extent*1e2, xmax=extent*1e2, ymax=extent*1e2)
        plt.title(r"Time: %.0f ps" % (time * 1e12) )
        plt.xlabel(r"$x$ (cm)")
        plt.ylabel(r"$y$ (cm)")

        plt.savefig("frame%03i.png" % i)

        advancer.advanceB(timeStep)
        advancer.advanceE(timeStep)    
        time += timeStep