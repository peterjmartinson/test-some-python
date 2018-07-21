import unittest
from BreastCancerML import *

class TestBreastCancerPredictionMachine(unittest.TestCase):

    # The input array needs 31 elements
    test_set = np.array([0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744, 0.4956, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173, 1.156, 3.445, 11.42, 14.91, 20.38, 26.5, 27.23, 77.58, 98.87, 386.1, 567.7, 84348301])

    def test_read_csv(self):
        data = pd.read_csv('Data/data.csv', index_col=False)
        self.assertFalse(data.empty)

    def test_pumper_NoneInput(self):
        pumper = BreastCancerPredictionMachine()
        input_data = None
        result = pumper.getData(input_data)
        self.assertEqual(result, 0)

    def test_pumper_SomeInput(self):
        pumper = BreastCancerPredictionMachine()
        input_data = 'data.csv'
        result = pumper.getData(input_data)
        self.assertEqual(result, 1)

    def test_pumper_NotCsv(self):
        pumper = BreastCancerPredictionMachine()
        input_data = 'data'
        result = pumper.getData(input_data)
        self.assertEqual(result, 0)

    def test_setInput_Exists(self):
        machine = BreastCancerPredictionMachine()
        result = machine.setInput(test_set)
        self.assertIsNotNone(result)

    def test_setInput_Returns_1_On_Success(self):
        machine = BreastCancerPredictionMachine()
        result = machine.setInput(test_set)
        self.assertEqual(result, 1)

if __name__ == '__main__':
    unittest.main()



