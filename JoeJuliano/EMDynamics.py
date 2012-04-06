import numpy as np
from scipy import weave
from scipy.weave import converters

speedOfLight = 299792458.0
epsilon0 = 8.854187817620e-12

class EMFields(object):

    def __init__(self, dim, edgeLength):
        self.edgeLength = edgeLength
        self.faceArea = edgeLength**2
        self.cellVolume = edgeLength**3
        self.dim = dim + (3,)

        maxEX = edgeLength * 0.5 * dim[0]
        self.eX = np.linspace(-maxEX, maxEX, dim[0], endpoint=False) \
            .reshape([dim[0],1,1])
        maxEY = edgeLength * 0.5 * dim[1]
        self.eY = np.linspace(-maxEY, maxEY, dim[1], endpoint=False) \
            .reshape([1,dim[1],1])
        maxEZ = edgeLength * 0.5 * dim[1]
        self.eZ = np.linspace(-maxEZ, maxEZ, dim[2], endpoint=False) \
            .reshape([1,1,dim[1]])

        self.bX = self.eX + 0.5 * edgeLength
        self.bY = self.eY + 0.5 * edgeLength
        self.bZ = self.eZ + 0.5 * edgeLength

        self.efield = np.zeros(self.dim, np.double)
        self.bfield = np.zeros(self.dim, np.double)
        self.current = np.zeros(self.dim, np.double)
        
    def setEField(self, location, value):
        self.efield[location] = value
        
    def getEField(self, location):
        return self.efield[location]

    def setBField(self, location, value):
        self.bfield[location] = value
        
    def getBField(self, location):
        return self.bfield[location]

    def setCurrent(self, location, value):
        self.current[location] = value
        
    def getCurrent(self, location):
        return self.current[location]


class EMDynamics:

    def __init__(self, fields):
        self.efield = fields.efield
        self.bfield = fields.bfield
        self.current = fields.current
        self.efieldRateFromB = 1.0 * speedOfLight**2 / fields.edgeLength
        self.efieldRateFromJ = 1.0 / (epsilon0 * fields.edgeLength)
        self.bfieldRate = 1.0 / fields.edgeLength 


class PurePythonDynamics(EMDynamics):
    
    def __init__(self, fields):
        EMDynamics.__init__(self, fields)

    def advanceB(self, deltaT):
        deltaB = deltaT * self.bfieldRate
        
        self.bfield[:, :, :, 1] -= \
            deltaB * np.roll(np.roll(self.efield[:, :, :, 0], -1, axis=1), -1, axis=2)
        self.bfield[:, :, :, 2] -= \
            deltaB * np.roll(self.efield[:, :, :, 0], -1, axis=2)
        self.bfield[:, :, :, 1] += \
            deltaB * np.roll(self.efield[:, :, :, 0], -1, axis=1)
        self.bfield[:, :, :, 2] += \
            deltaB * np.roll(np.roll(self.efield[:, :, :, 0], -1, axis=1), -1, axis=2)
        
        self.bfield[:, :, :, 0] += \
            deltaB * np.roll(np.roll(self.efield[:, :, :, 1], -1, axis=0), -1, axis=2)
        self.bfield[:, :, :, 2] += \
            deltaB * np.roll(self.efield[:, :, :, 1], -1, axis=2)
        self.bfield[:, :, :, 0] -= \
            deltaB * np.roll(self.efield[:, :, :, 1], -1, axis=0)
        self.bfield[:, :, :, 2] -= \
            deltaB * np.roll(np.roll(self.efield[:, :, :, 1], -1, axis=0), -1, axis=2)

        self.bfield[:, :, :, 0] -= \
            deltaB * np.roll(np.roll(self.efield[:, :, :, 2], -1, axis=0), -1, axis=1)
        self.bfield[:, :, :, 1] -= \
            deltaB * np.roll(self.efield[:, :, :, 2], -1, axis=1)
        self.bfield[:, :, :, 0] += \
            deltaB * np.roll(self.efield[:, :, :, 2], -1, axis=0)
        self.bfield[:, :, :, 1] += \
            deltaB * np.roll(np.roll(self.efield[:, :, :, 2], -1, axis=0), -1, axis=1)

    def advanceE(self, deltaT):
        deltaEB = deltaT * self.efieldRateFromB
        deltaEJ = deltaT * self.efieldRateFromJ
        
        self.efield[:, :, :, 1] += \
            deltaEB * np.roll(self.bfield[:, :, :, 0], 1, axis=0)
        self.efield[:, :, :, 2] += \
            deltaEB * np.roll(np.roll(self.bfield[:, :, :, 0], 1, axis=0), 1, axis=1)
        self.efield[:, :, :, 1] -= \
            deltaEB * np.roll(np.roll(self.bfield[:, :, :, 0], 1, axis=0), 1, axis=2)
        self.efield[:, :, :, 2] -= \
            deltaEB * np.roll(self.bfield[:, :, :, 0], 1, axis=0)

        self.efield[:, :, :, 0] -= \
            deltaEB * np.roll(self.bfield[:, :, :, 1], 1, axis=1)
        self.efield[:, :, :, 2] -= \
            deltaEB * np.roll(np.roll(self.bfield[:, :, :, 1], 1, axis=1), 1, axis=0)
        self.efield[:, :, :, 0] += \
            deltaEB * np.roll(np.roll(self.bfield[:, :, :, 1], 1, axis=1), 1, axis=2)
        self.efield[:, :, :, 2] += \
            deltaEB * np.roll(self.bfield[:, :, :, 1], 1, axis=1)

        self.efield[:, :, :, 0] += \
            deltaEB * np.roll(self.bfield[:, :, :, 2], 1, axis=2)
        self.efield[:, :, :, 1] += \
            deltaEB * np.roll(np.roll(self.bfield[:, :, :, 2], 1, axis=2), 1, axis=0)
        self.efield[:, :, :, 0] -= \
            deltaEB * np.roll(np.roll(self.bfield[:, :, :, 2], 1, axis=2), 1, axis=1)
        self.efield[:, :, :, 1] -= \
            deltaEB * np.roll(self.bfield[:, :, :, 2], 1, axis=2)

        self.efield[:, :, :, :] -= deltaEJ * self.current[:, :, :, :]

    
class WeaveCDynamics(EMDynamics):
    
    def __init__(self, fields):
        EMDynamics.__init__(self, fields)

    def advanceB(self, deltaT):
        deltaB = deltaT * self.bfieldRate
        efield = self.efield
        bfield = self.bfield
        shape = np.array(efield.shape)

        code = """
        for (int i=0; i<shape(0); ++i) {
            int inext = (i + 1) % shape(0);
            for (int j=0; j<shape(1); ++j) {
                int jnext = (j + 1) % shape(1);
                for (int k=0; k<shape(2); ++k) {
                    int knext = (k + 1) % shape(2);
                    
                    bfield(i,j,k,1) -= deltaB * efield(i,jnext,knext,0);
                    bfield(i,j,k,2) -= deltaB * efield(i,j,knext,0);
                    bfield(i,j,k,1) += deltaB * efield(i,jnext,k,0);
                    bfield(i,j,k,2) += deltaB * efield(i,jnext,knext,0);
                    
                    bfield(i,j,k,0) += deltaB * efield(inext,j,knext,1);
                    bfield(i,j,k,2) += deltaB * efield(i,j,knext,1);
                    bfield(i,j,k,0) -= deltaB * efield(inext,j,k,1);
                    bfield(i,j,k,2) -= deltaB * efield(inext,j,knext,1);
                    
                    bfield(i,j,k,0) -= deltaB * efield(inext,jnext,k,2);
                    bfield(i,j,k,1) -= deltaB * efield(i,jnext,k,2);
                    bfield(i,j,k,0) += deltaB * efield(inext,j,k,2);
                    bfield(i,j,k,1) += deltaB * efield(inext,jnext,k,2);
                }
            }
        }
        """
        weave.inline(code, ['efield', 'bfield', 'shape', 'deltaB'],
                     type_converters=converters.blitz)


    def advanceE(self, deltaT):
        deltaE = deltaT * self.efieldRate
        efield = self.efield
        bfield = self.bfield
        shape = np.array(efield.shape)

        code = """
        for (int i=0; i<shape(0); ++i) {
            int iprev = (i + shape(0) - 1) % shape(0);
            for (int j=0; j<shape(1); ++j) {
                int jprev = (j + shape(1) - 1) % shape(1);
                for (int k=0; k<shape(2); ++k) {
                    int kprev = (k + shape(2) - 1) % shape(2);
                    
                    efield(i,j,k,1) += deltaE * bfield(iprev,j,k,0);
                    efield(i,j,k,2) += deltaE * bfield(iprev,jprev,k,0);
                    efield(i,j,k,1) -= deltaE * bfield(iprev,j,kprev,0);
                    efield(i,j,k,2) -= deltaE * bfield(iprev,j,k,0);

                    efield(i,j,k,0) -= deltaE * bfield(i,jprev,k,1);
                    efield(i,j,k,2) -= deltaE * bfield(iprev,jprev,k,1);
                    efield(i,j,k,0) += deltaE * bfield(i,jprev,kprev,1);
                    efield(i,j,k,2) += deltaE * bfield(i,jprev,k,1);

                    efield(i,j,k,0) += deltaE * bfield(i,j,kprev,2);
                    efield(i,j,k,1) += deltaE * bfield(iprev,j,kprev,2);
                    efield(i,j,k,0) -= deltaE * bfield(i,jprev,kprev,2);
                    efield(i,j,k,1) -= deltaE * bfield(i,j,kprev,2);
                }
            }
        }
        """
        weave.inline(code, ['efield', 'bfield', 'shape', 'deltaE'],
                     type_converters=converters.blitz)
