import unittest
from BreastCancerML import *
from sklearn.preprocessing import StandardScaler

class TestBreastCancerPredictionMachine(unittest.TestCase):

    def test_setInput_Exists(self):
        machine = BreastCancerPredictionMachine()
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        result = machine.setInput(test_data)
        self.assertIsNotNone(result)

    def test_setInput_Must_Take_A_List(self):
        machine = BreastCancerPredictionMachine()
        test_data = 'Not an array'
        with self.assertRaises(TypeError):
          machine.setInput(test_data)

    def test_setInput_Does_Not_Take_Less_Than_31_Elements(self):
        machine = BreastCancerPredictionMachine()
        test_data = [1, 2, 3]
        with self.assertRaises(ValueError):
          machine.setInput(test_data)

    def test_setInput_Does_Not_Take_More_Than_31_Elements(self):
        machine = BreastCancerPredictionMachine()
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        with self.assertRaises(ValueError):
          machine.setInput(test_data)

    def test_setInput_Returns_Numpy_Array(self):
        machine = BreastCancerPredictionMachine()
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        result = machine.setInput(test_data)
        self.assertEqual(result.size, 31)

    def test_getDiagnosis_Returns_Single_Character_On_Success(self):
        machine = BreastCancerPredictionMachine()
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        result = machine.getDiagnosis(test_data)
        self.assertEqual(len(result), 1)

    def test_getDiagnosis_Raises_TypeError_Exception_If_Input_Is_Not_List_Or_Numpy_Array(self):
        machine = BreastCancerPredictionMachine()
        test_data = "not a list or Numpy array"
        with self.assertRaises(TypeError):
            machine.getDiagnosis(test_data)

    def test_getDiagnosis_Predicts_Malignant_Correctly(self):
        machine = BreastCancerPredictionMachine()
        test_data = [0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744, 0.4956, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173, 1.156, 3.445, 11.42, 14.91, 20.38, 26.5, 27.23, 77.58, 98.87, 386.1, 567.7, 84348301]
        result = machine.getDiagnosis(test_data)
        self.assertEqual(result, 'M')
    
    def test_getDiagnosis_Predicts_Benign_Correctly(self):
        machine = BreastCancerPredictionMachine()
        test_data = [8510426,13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
        result = machine.getDiagnosis(test_data)
        self.assertEqual(result, 'B')

    def test_getScaler(self):
        machine = BreastCancerPredictionMachine()
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        test_sample_data = machine.setInput(test_data)
        result = machine.getScaler(test_sample_data)
        # self.assertEqual(type(result), sklearn.preprocessing.data.StandardScaler)
        assert type(result) != list




if __name__ == '__main__':
    unittest.main()



