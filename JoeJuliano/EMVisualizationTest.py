import unittest
import EMVisualization
import EMDynamics
import numpy as np
from numpy.testing.utils import assert_allclose

class DirectVisualizationTest(unittest.TestCase):

    def setUp(self):
        self.shape = (4, 4, 4)
        self.edgeLength = 1e-2
        self.fields = EMDynamics.EMFields(self.shape, self.edgeLength)
        self.fields.efield[2,2,2,0] = 1.0
        self.fields.efield[2,2,2,1] = 2.0
        self.fields.bfield[2,2,2,0] = 3.0
        self.fields.bfield[2,2,2,1] = 4.0
        self.visualization = EMVisualization.DirectVisualization(self.fields)
        self.grid = np.zeros((4, 4))


    def testEXX(self):
        expected = np.linspace(-0.015, 0.015, 4).reshape([4,1,1])
        assert_allclose(self.visualization.EXX, expected, 1e-14)

    def testEXY(self):
        expected = np.linspace(-0.02, 0.01, 4).reshape([1,4,1])
        assert_allclose(self.visualization.EXY, expected, 1e-14)

    def testEYX(self):
        expected = np.linspace(-0.02, 0.01, 4).reshape([4,1,1])
        assert_allclose(self.visualization.EYX, expected, 1e-14)

    def testEYY(self):
        expected = np.linspace(-0.015, 0.015, 4).reshape([1,4,1])
        assert_allclose(self.visualization.EYY, expected, 1e-14)
        
    def testEZX(self):
        expected = np.linspace(-0.02, 0.01, 4).reshape([4,1,1])
        assert_allclose(self.visualization.EZX, expected, 1e-14)
        
    def testEZY(self):
        expected = np.linspace(-0.02, 0.01, 4). reshape([1,4,1])
        assert_allclose(self.visualization.EZY, expected, 1e-4)

    def testBXX(self):
        expected = np.linspace(-0.01, 0.02, 4).reshape([4,1,1])
        assert_allclose(self.visualization.BXX, expected, 1e-14)

    def testBXY(self):
        expected = np.linspace(-0.015, 0.015, 4).reshape([1,4,1])
        assert_allclose(self.visualization.BXY, expected, 1e-14)

    def testBYX(self):
        expected = np.linspace(-0.015, 0.015, 4).reshape([4,1,1])
        assert_allclose(self.visualization.BYX, expected, 1e-14)

    def testBYY(self):
        expected = np.linspace(-0.01, 0.02, 4).reshape([1,4,1])
        assert_allclose(self.visualization.BYY, expected, 1e-14)

    def testBZX(self):
        expected = np.linspace(-0.015, 0.015, 4).reshape([4,1,1])
        assert_allclose(self.visualization.BZX, expected, 1e-14)
        
    def testBZY(self):
        expected = np.linspace(-0.015, 0.015, 4). reshape([1,4,1])
        assert_allclose(self.visualization.BZY, expected, 1e-4)

        
    def testEX(self):
        self.visualization.calculate()
        self.grid[2,2] = 1.0
        assert_allclose(self.visualization.EX, self.grid, 1e-14)
        
    def testEY(self):
        self.visualization.calculate()
        self.grid[2,2] = 2.0
        assert_allclose(self.visualization.EY, self.grid, 1e-14)
        
    def testBX(self):
        self.visualization.calculate()
        self.grid[2,2] = 3.0 * EMDynamics.speedOfLight
        assert_allclose(self.visualization.BX, self.grid, 1e-14)
        
    def testBY(self):
        self.visualization.calculate()
        self.grid[2,2] = 4.0 * EMDynamics.speedOfLight
        assert_allclose(self.visualization.BY, self.grid, 1e-14)
        
if __name__ == "__main__":
    unittest.main()