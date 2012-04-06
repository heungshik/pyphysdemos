# -*- coding: utf-8 -*-
import EMDynamics
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import EMVisualization

if __name__ == '__main__':
    print "EM example “demo 3”"
    

    edgeLength = 0.01
    timeStep = 0.4 * edgeLength / EMDynamics.speedOfLight
    N = 40
    extent = edgeLength * N * 0.5
    
    fields = EMDynamics.EMFields((N,N,N), edgeLength)
    advancer = EMDynamics.WeaveCDynamics(fields)
#    advancer = EMDynamics.PurePythonDynamics(fields)
    
    viz = EMVisualization.DirectVisualization(fields)

    #Set up a plane wave.
    kx = 6 * (2 * math.pi / (edgeLength * N))
    emax = 1.0
    bmax = emax / EMDynamics.speedOfLight
    
    fields.efield[:,:,:,1] = emax * np.sin(kx * fields.eX)
    fields.bfield[:,:,:,2] = bmax * np.sin(kx * fields.bX)
    
    alpha = 2.0 / edgeLength
    envelope = np.exp(-alpha * 
                      ((fields.eX+0.5*extent)**2 + fields.eY**2 + fields.eZ**2))
    envelope = envelope.reshape([N,N,N,1])
    fields.efield *= envelope
    fields.bfield *= envelope
        
    scale = 40.0
    
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
