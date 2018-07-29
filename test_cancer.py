import unittest
from BreastCancerML import *

class TestBreastCancerPredictionMachine_setInput(unittest.TestCase):

    def setUp(self):
        self.machine = BreastCancerPredictionMachine()

    def test_setInput_Exists(self):
        self.assertTrue(hasattr(self.machine, 'setInput'))

    def test_setInput_Must_Take_A_List(self):
        test_data = 'Not an array'
        with self.assertRaises(TypeError):
          self.machine.setInput(test_data)

    def test_setInput_Does_Not_Take_Less_Than_31_Elements(self):
        test_data = [i + 1 for i in range(3)]
        with self.assertRaises(ValueError):
          self.machine.setInput(test_data)

    def test_setInput_Does_Not_Take_More_Than_31_Elements(self):
        test_data = [i + 1 for i in range(41)]
        with self.assertRaises(ValueError):
          self.machine.setInput(test_data)

    def test_setInput_Returns_Numpy_Array(self):
        test_data = [i + 1 for i in range(31)]
        result = self.machine.setInput(test_data)
        self.assertEqual(result.size, 31)

class TestBreastCancerPredictionMachine_getDiagnosis(unittest.TestCase):

    def setUp(self):
        self.machine = BreastCancerPredictionMachine()

    def test_getDiagnosis_Exists(self):
        self.assertTrue(hasattr(self.machine, 'getDiagnosis'))

    def test_getDiagnosis_Returns_Single_Character_On_Success(self):
        test_data = [i + 1 for i in range(31)]
        result = self.machine.getDiagnosis(test_data)
        self.assertEqual(len(result), 1)

    def test_getDiagnosis_Raises_TypeError_Exception_If_Input_Is_Not_List_Or_Numpy_Array(self):
        test_data = "not a list or Numpy array"
        with self.assertRaises(TypeError):
            self.machine.getDiagnosis(test_data)

    def test_getDiagnosis_Predicts_Malignant_Correctly(self):
        test_data = [0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744, 0.4956, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173, 1.156, 3.445, 11.42, 14.91, 20.38, 26.5, 27.23, 77.58, 98.87, 386.1, 567.7, 84348301]
        result = self.machine.getDiagnosis(test_data)
        self.assertEqual(result, 'M')
    
    def test_getDiagnosis_Predicts_Benign_Correctly(self):
        test_data = [8510426,13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
        result = self.machine.getDiagnosis(test_data)
        self.assertEqual(result, 'B')

class TestBreastCancerPredictionMachine_train(unittest.TestCase):
    
    def setUp(self):
        self.machine = BreastCancerPredictionMachine()

    def test_train_Exists(self):
        self.assertTrue(hasattr(self.machine, 'train'))

    def test_train_Takes_A_Pandas_DataFrame(self):
        training_data = "not a Pandas DataFrame"
        with self.assertRaises(TypeError):
            self.machine.train(training_data)
      



if __name__ == '__main__':
    unittest.main()



