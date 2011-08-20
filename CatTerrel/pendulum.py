#!/usr/bin/env python
import math, numpy, matplotlib.pyplot as plt

print "\nCalculate a Trajectory"

#Setting Conditions
tfinal = 10
dt = 0.1
t = numpy.arange(0,tfinal + dt, dt)
nstep = len(t)-1

mass = 1.0
g = 9.8
L = 5.0

#Initial Conditions
theta = numpy.zeros(nstep+1)
omega = numpy.zeros(nstep+1)
alpha = numpy.zeros(nstep+1)

theta[1] = 0.03
omega[1] = 0.1
alpha[0] = 0

#Defining a
def a(theta):
    return -3*g/(2*L)*numpy.sin(theta)

#Defining initial theta
theta[0] = theta[1] - dt * omega[1] + dt * dt * a(theta[1])/2

#Defining theta, omega in terms of i, Verlet method
for i in range(1,nstep):
    theta[i+1] = 2 * theta[i] - theta[i-1] + dt * dt * alpha[i]
    omega[i] = (theta[i+1] - theta[i-1])/(2 * dt)

    alpha[i+1] = a(theta[i+1])
    
omega[0] = (4*theta[1]-theta[2]-3*theta[0])/(2 * dt)
omega[nstep] = (4*theta[nstep-1]-theta[nstep-2]-3*theta[nstep])/(2 * dt)

plt.plot(theta, omega,'ro')
plt.xlabel("theta")
plt.ylabel("angular velocity")
plt.grid(True)
plt.savefig("Pendulum2.pdf")

plt.clf()
plt.plot(t, theta,'ro')
plt.xlabel("t")
plt.ylabel("theta")
plt.grid(True)
plt.savefig("Pendulum3.pdf")
