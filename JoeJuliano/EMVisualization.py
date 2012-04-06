import numpy as np
import EMDynamics

class DirectVisualization:
    
    def __init__(self, fields):
        self.fields = fields
        self.eLevel = fields.dim[2]/2
        self.bLevel = fields.dim[2]/2
        
        edgeLength = fields.edgeLength
        halfEdgeLength = edgeLength * 0.5
        
        self.EXX = fields.eX + halfEdgeLength
        self.EXY = fields.eY
        self.EYX = fields.eX
        self.EYY = fields.eY + halfEdgeLength
        self.EZX = fields.eX
        self.EZY = fields.eY

        self.BXX = fields.bX + halfEdgeLength
        self.BXY = fields.bY
        self.BYX = fields.bX
        self.BYY = fields.bY + halfEdgeLength
        self.BZX = fields.bX
        self.BZY = fields.bY
        
        self.EX = np.zeros(fields.dim[:2])
        self.EY = np.zeros(fields.dim[:2])
        self.EZ = np.zeros(fields.dim[:2])
        self.BX = np.zeros(fields.dim[:2])
        self.BY = np.zeros(fields.dim[:2])
        self.BZ = np.zeros(fields.dim[:2])
        self.zeros = np.zeros(fields.dim[:2])
        
            
    def calculate(self):
        self.EX[:,:] = self.fields.efield[:,:,self.eLevel,0].transpose()
        self.EY[:,:] = self.fields.efield[:,:,self.eLevel,1].transpose()
        self.EZ[:,:] = self.fields.efield[:,:,self.eLevel,2].transpose()
        self.BX[:,:] = self.fields.bfield[:,:,self.bLevel,0].transpose() \
            * EMDynamics.speedOfLight
        self.BY[:,:] = self.fields.bfield[:,:,self.bLevel,1].transpose() \
            * EMDynamics.speedOfLight
        self.BZ[:,:] = self.fields.bfield[:,:,self.bLevel,2].transpose() \
            * EMDynamics.speedOfLight