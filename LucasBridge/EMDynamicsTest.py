import unittest

import EMDynamics
import numpy as np
from numpy.testing.utils import assert_allclose
from numpy.ma.testutils import assert_equal

class EMFieldsTest(unittest.TestCase):

    def setUp(self):
        self.shape = (4, 4, 4)
        self.edgeLength = 1e-2
        self.fields = EMDynamics.EMFields(self.shape, self.edgeLength)
        self.advancer = EMDynamics.PurePythonDynamics(self.fields)
    
    def testFieldArrayShape(self):
        assert_equal(self.fields.efield.shape, self.shape + (3,))
        assert_equal(self.fields.bfield.shape, self.shape + (3,))

    def testSetEField(self):
        self.fields.setEField((0, 0, 0), (1, 2, 3))
        value = self.fields.getEField((0, 0, 0))
        assert_allclose(value, (1.0, 2.0, 3.0))
        
    def testSetBField(self):
        self.fields.setBField((0, 0, 0), (2, 3, 1))
        value = self.fields.getBField((0, 0, 0))
        assert_allclose(value, (2.0, 3.0, 1.0))
    
    def testSetCurrent(self):
        self.fields.setCurrent((0,0,0), (3, 1, 2))
        value = self.fields.getCurrent((0, 0, 0))
        assert_allclose(value, (3.0, 1.0, 2.0))

    def testFaceArea(self):
        assert_allclose(self.fields.faceArea, self.edgeLength**2)

    def testCellVolume(self):
        assert_allclose(self.fields.cellVolume, self.edgeLength**3)
        
    def testEX(self):
        expect = np.array([-0.02, -0.01, 0.00, 0.01]).reshape([4,1,1])
        assert_allclose(self.fields.eX, expect, 1e-12)

    def testEY(self):
        expect = np.array([-0.02, -0.01, 0.00, 0.01]).reshape([1,4,1])
        assert_allclose(self.fields.eY, expect, 1e-12)

    def testEZ(self):
        expect = np.array([-0.02, -0.01, 0.00, 0.01]).reshape([1,1,4])
        assert_allclose(self.fields.eZ, expect, 1e-12)

    def testBX(self):
        expect = np.array([-0.015, -0.005, 0.005, 0.015]).reshape([4,1,1])
        assert_allclose(self.fields.bX, expect, 1e-12)

    def testBY(self):
        expect = np.array([-0.015, -0.005, 0.005, 0.015]).reshape([1,4,1])
        assert_allclose(self.fields.bY, expect, 1e-12)

    def testBZ(self):
        expect = np.array([-0.015, -0.005, 0.005, 0.015]).reshape([1,1,4])
        assert_allclose(self.fields.bZ, expect, 1e-12)

class AdvancerTestBase:

    def initialize(self, shape, edgeLength):
        self.shape = shape
        self.edgeLength = edgeLength
        self.fields = EMDynamics.EMFields(self.shape, self.edgeLength)
        self.deltaT = 0.5*edgeLength/EMDynamics.speedOfLight
        self.deltaB = self.deltaT / self.edgeLength
        self.deltaEFromB = \
            self.deltaT * EMDynamics.speedOfLight**2 / self.edgeLength
        self.deltaEFromJ = \
            self.deltaT / (EMDynamics.epsilon0 * self.edgeLength)

    def recordFields(self):
        expectE = np.zeros(self.shape + (3,))
        expectB = np.zeros(self.shape + (3,))
        expectE[:, :, :, :] = self.fields.efield
        expectB[:, :, :, :] = self.fields.bfield
        return expectE, expectB

    def testAdvanceEmptyField(self):
        expectE, expectB = self.recordFields()
        self.advancer.advanceB(self.deltaT)
        self.advancer.advanceE(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)
        
    def testAdvanceOneEXVector(self):
        self.fields.setEField((1, 1, 1), (1.0, 0.0, 0.0))
        expectE, expectB = self.recordFields()
        expectB[1, 0, 0, 1] -= self.deltaB
        expectB[1, 1, 0, 2] -= self.deltaB
        expectB[1, 0, 1, 1] += self.deltaB
        expectB[1, 0, 0, 2] += self.deltaB
        self.advancer.advanceB(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneEYVector(self):
        self.fields.setEField((1, 1, 1), (0.0, 1.0, 0.0))
        expectE, expectB = self.recordFields()
        expectB[0, 1, 0, 0] += self.deltaB
        expectB[1, 1, 0, 2] += self.deltaB
        expectB[0, 1, 1, 0] -= self.deltaB
        expectB[0, 1, 0, 2] -= self.deltaB
        self.advancer.advanceB(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneEZVector(self):
        self.fields.setEField((1, 1, 1), (0.0, 0.0, 1.0))
        expectE, expectB = self.recordFields()
        expectB[0, 0, 1, 0] -= self.deltaB
        expectB[1, 0, 1, 1] -= self.deltaB
        expectB[0, 1, 1, 0] += self.deltaB
        expectB[0, 0, 1, 1] += self.deltaB
        self.advancer.advanceB(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneBXVector(self):
        self.fields.setBField((1, 1, 1), (1.0, 0.0, 0.0))
        expectE, expectB = self.recordFields()
        expectE[2, 1, 1, 1] += self.deltaEFromB
        expectE[2, 2, 1, 2] += self.deltaEFromB
        expectE[2, 1, 2, 1] -= self.deltaEFromB
        expectE[2, 1, 1, 2] -= self.deltaEFromB
        self.advancer.advanceE(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneBYVector(self):
        self.fields.setBField((1, 1, 1), (0.0, 1.0, 0.0))
        expectE, expectB = self.recordFields()
        expectE[1, 2, 1, 0] -= self.deltaEFromB
        expectE[2, 2, 1, 2] -= self.deltaEFromB
        expectE[1, 2, 2, 0] += self.deltaEFromB
        expectE[1, 2, 1, 2] += self.deltaEFromB
        self.advancer.advanceE(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneBZVector(self):
        self.fields.setBField((1, 1, 1), (0.0, 0.0, 1.0))
        expectE, expectB = self.recordFields()
        expectE[1, 1, 2, 0] += self.deltaEFromB
        expectE[2, 1, 2, 1] += self.deltaEFromB
        expectE[1, 2, 2, 0] -= self.deltaEFromB
        expectE[1, 1, 2, 1] -= self.deltaEFromB
        self.advancer.advanceE(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneJXVector(self):
        self.fields.setCurrent((1, 1, 1), (1.0, 0.0, 0.0))
        expectE, expectB = self.recordFields()
        expectE[1, 1, 1, 0] -= self.deltaEFromJ
        self.advancer.advanceE(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneJYVector(self):
        self.fields.setCurrent((1, 1, 1), (0.0, 1.0, 0.0))
        expectE, expectB = self.recordFields()
        expectE[1, 1, 1, 1] -= self.deltaEFromJ
        self.advancer.advanceE(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

    def testAdvanceOneJZVector(self):
        self.fields.setCurrent((1, 1, 1), (0.0, 0.0, 1.0))
        expectE, expectB = self.recordFields()
        expectE[1, 1, 1, 2] -= self.deltaEFromJ
        self.advancer.advanceE(self.deltaT)
        assert_allclose(self.fields.efield, expectE)
        assert_allclose(self.fields.bfield, expectB)

class PurePythonDynamicsTest(unittest.TestCase, AdvancerTestBase):

    def setUp(self):
        self.initialize((4,4,4), 1e-2)
        self.advancer = EMDynamics.PurePythonDynamics(self.fields)


#class WeaveCDynamicsTest(unittest.TestCase, AdvancerTestBase):
#
#    def setUp(self):
#        self.initialize((4,4,4), 1e-2)
#        self.advancer = EMDynamics.WeaveCDynamics(self.fields)


if __name__ == "__main__":
    try: 
        unittest.main()
    except SystemExit:
        pass
