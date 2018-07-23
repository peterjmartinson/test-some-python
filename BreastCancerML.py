import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time

class BreastCancerPredictionMachine(object):

    def setInput(self, input_list):
        if type(input_list) != list:
            raise TypeError('input must be a list')
        if len(input_list) != 31:
            raise ValueError('input must be a list with 31 elements')
        numpy_array = np.array(input_list).astype(np.float64)
        return numpy_array

    def getDiagnosis(self, data):
        if type(data) != list and type(data) != np.ndarray:
            raise TypeError('input must be a list or Numpy array')
        # Train the model
        training_data = 'Data/data.csv'
        pandas_data = pd.read_csv(training_data, index_col=False)
        Y = pandas_data['diagnosis'].values # Get everything down the diagnosis column
        X = pandas_data.drop('diagnosis', axis=1).values # get everything *but* the diagnosis column
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X) ## normalize the X values (2d matrix)
        model = SVC(C=2.0, kernel='rbf') ## choose the model
        model.fit(X_scaled, Y) ## train the model.  (input values, answers)

        # Make the prediction
        if type(data) != np.ndarray:
            sample_data = self.setInput(data)
        else:
            sample_data = data

        sample_data_scaled = scaler.transform(sample_data.reshape(1,-1)) ## One line of values, normalized.  These are the test values
        predictions = model.predict(sample_data_scaled) ## Output - what should these values give you?

        return predictions[0]

    def getScaler(self, sample_data):
        return 1

    
  


input_data = 'Data/data.csv'
data = pd.read_csv(input_data, index_col=False)

Y = data['diagnosis'].values # Get everything down the diagnosis column
X = data.drop('diagnosis', axis=1).values # get everything *but* the diagnosis column
print(type(X))

machine = BreastCancerPredictionMachine()
result_1 = machine.getDiagnosis(X[10,:])
result_2 = machine.getDiagnosis(X[20,:])

print("Predicted: {}, actual: {}".format(result_1, Y[10]));
print("Predicted: {}, actual: {}".format(result_2, Y[20]));

