import numpy as np, math, matplotlib.pyplot as plt

class Molecule:
    def __init__(self,natom):
        print "initializing molecule"
        self.natom=natom
        self.r=np.zeros((self.natom, 3))
        self.mass=np.zeros(self.natom)
        self.AMU=1.66054E-24 #AMU in grams
        self.delta=1E-18
        self.dr=np.zeros((self.natom, 3))
    def v(self, dr):
        v = 0.
        return v
    def calcDynamicMatrix(self):
        self.mat = np.zeros([self.natom*3,self.natom*3])
        for iatom in xrange(2):
            for idim in xrange(3):
                for jatom in xrange(2):
                    for jdim in xrange(3):
                        sum=0.
                        self.dr[:,:] = 0.
                        self.dr[iatom,idim] += self.delta
                        self.dr[jatom,jdim] += self.delta
                        sum+= self.v(self.dr)
                        self.dr[jatom,jdim] -= 2*self.delta
                        sum-= self.v(self.dr)
                        self.dr[iatom,idim] -= 2*self.delta
                        self.dr[jatom,jdim] += 2*self.delta
                        sum-= self.v(self.dr)
                        self.dr[jatom,jdim] -= 2*self.delta
                        sum+= self.v(self.dr)
                        self.dr[:,:] = 0.
                        sum /= 4*self.delta**2
                        self.mat[3*iatom+idim,3*jatom+jdim]= \
                          sum/math.sqrt(self.mass[iatom]*self.mass[jatom])
    def calcMolecularVibrations(self):
        momega2,self.mode = np.linalg.eig(self.mat)
        self.omega = np.sqrt(np.abs(momega2))*np.sign(momega2)
        self.freq = self.omega/(2*math.pi)

class H2(Molecule):
    def __init__(self):
        Molecule.__init__(self,2)
        print "initializing H2"
        self.a=19.3*1E7
        self.v0=109.457*6.948E-14
        self.d0=0.07414*1E-7
        #set the atom positions
        self.r[0,0]=-self.d0*0.5
        self.r[1,0]=self.d0*0.5
        self.mass[:]=1.0079*self.AMU
    def v(self, dr):
        rvec = (self.r[1,:]+dr[1,:])-(self.r[0,:]+dr[0,:])
        r = math.sqrt(np.dot(rvec,rvec))
        v=self.v0*(1.-math.exp(-self.a*math.fabs(r-self.d0)))**2
        return v
    
class H2O(Molecule):
    def __init__(self):
        Molecule.__init__(self,3)
        print "initializing H20"
        self.k1=85.54E4
        self.k2=-1.01E4
        self.k3=2.288E4
        self.k4=7.607E4
        self.r0=0.9572E-8
        self.theta0=107*math.pi/180.
        #set the atom positions
        self.r[1,0]=self.r0*math.sin(self.theta0*0.5)
        self.r[1,1]=-self.r0*math.cos(self.theta0*0.5)
        self.r[2,0]=-self.r0*math.sin(self.theta0*0.5)
        self.r[2,1]=-self.r0*math.cos(self.theta0*0.5)
        self.mass[0]=15.9994*self.AMU
        self.mass[1:]=1.0079*self.AMU
    def v(self, dr):
        rvec1 = (self.r[1,:]+dr[1,:])-(self.r[0,:]+dr[0,:])
        r1 = math.sqrt(np.dot(rvec1,rvec1))
        dr1=r1-self.r0
        rvec2 = (self.r[2,:]+dr[2,:])-(self.r[0,:]+dr[0,:])
        r2 = math.sqrt(np.dot(rvec2,rvec2))
        dr2=r2-self.r0
        theta=math.acos(np.dot(rvec1,rvec2)/(r1*r2))
        rdtheta=self.r0*(theta-self.theta0)
        v=0.5*self.k1*(dr1*dr1+dr2*dr2)\
           +self.k2*dr1*dr2\
           +self.k3*(dr1+dr2)*rdtheta\
           +0.5*self.k4*(rdtheta*rdtheta)
        return v


# Function to make a plot.
def makePlot(mol, filename):
    for iatom in xrange(mol.natom):
        for idim in xrange(3):
            imode = iatom*3+idim
            print mol.mode[:,imode]
            plt.subplot(3,mol.natom,imode+1)
            plt.plot(mol.r[:,0]*1e8,mol.r[:,1]*1e8,'r.')
            plt.axis(xmin=-1,xmax=1,ymin=-1,ymax=1)
            plt.quiver(mol.r[:,0]*1e8,mol.r[:,1]*1e8,
                       mol.mode[0:-1:3,imode],mol.mode[1:-1:3,imode])
            plt.text(0,-0.8,r"$\nu = %.0f$ cm$^{-1}$" % (mol.freq[imode]/2.99782458e10),
                     ha="center", va="center")
    plt.savefig(filename)


h2=H2()
h2.calcDynamicMatrix()
print "Dyanamical matrix is \n", h2.mat
h2.calcMolecularVibrations()
print h2.freq, "Frequency in Hz"
print h2.freq/2.99782458E10, "Frequency in cm-1"
makePlot(h2,"h2vibration.pdf")



h2o=H2O()
h2o.v(h2o.dr)
h2o.calcDynamicMatrix()
print "Dyanamical matrix is \n", h2o.mat
h2o.calcMolecularVibrations()
print h2o.freq, "Frequency in Hz"
print h2o.freq/2.99782458E10, "Frequency in cm-1"
makePlot(h2o,"h2ovibration.pdf")
