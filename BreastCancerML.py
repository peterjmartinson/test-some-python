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

    def setInput(self, input_data):
        if len(input_data) != 31:
            raise ValueError('input must be a list with 31 elements')
        numpy_array = np.array(input_data)
        return numpy_array

    def getDiagnosis(self, data):
        if type(data) is not list:
            raise TypeError('input must be a list')
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
        sample_data = self.setInput(data)
        sample_data_scaled = scaler.transform(sample_data.reshape(1,-1)) ## One line of values, normalized.  These are the test values
        predictions = model.predict(sample_data_scaled) ## Output - what should these values give you?
        return predictions[0]
        # return 'Mn'


  

# bring in the data with Pandas and print the first few lines

# input_data = 'Data/data.csv'
# data = pd.read_csv(input_data, index_col=False)
# print(data.head(5))

# assign data to the correct axes

# Y = data['diagnosis'].values # Get everything down the diagnosis column
# X = data.drop('diagnosis', axis=1).values # get everything *but* the diagnosis column


# test_set = np.array([0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744, 0.4956, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173, 1.156, 3.445, 11.42, 14.91, 20.38, 26.5, 27.23, 77.58, 98.87, 386.1, 567.7, 84348301])





# --------------------------------------------------------------------------------
## The below "is not needed"
# Below shows SVC is better than Logistics model; DO NOT NEED FOR PRODUCTION!
# num_folds = 10
# kfold = KFold(n_splits=num_folds, random_state=123)
# start = time.time()
# cv_results = cross_val_score(LogisticRegression(), X, Y, cv=kfold, scoring='accuracy')
# end = time.time()
# print( "Logistics regression accuracy: %f, run time: %f)" % (cv_results.mean(), end-start))

# start = time.time()
# scaler = StandardScaler().fit(X)
# X_scaled = scaler.transform(X)
# cv_results = cross_val_score(SVC(C=2.0, kernel="rbf"), X_scaled, Y, cv=kfold, scoring='accuracy')
# end = time.time()
# print( "SVC accuracy: %f, run time: %f)" % (cv_results.mean(), end-start))
## The above "is not needed"
# --------------------------------------------------------------------------------

## The below is the "training"
# THIS IS WHAT IS NEEDED FOR PRODUCTION TO ESTIMATE THE MODEL
# scaler = StandardScaler().fit(X)
# X_scaled = scaler.transform(X) ## normalize the X values (2d matrix)
# model = SVC(C=2.0, kernel='rbf') ## choose the model
# model.fit(X_scaled, Y) ## train the model.  (input values, answers)


# THIS IS WHAT IS NEEDED FOR PREDICTION
# X_test_scaled = scaler.transform(X[10,:].reshape(1,-1)) ## One line of values, normalized.  These are the test values
# predictions = model.predict(X_test_scaled) ## Output - what should these values give you?

# print("Predicted: {}, actual: {}".format(predictions[0], Y[10]));

# X_test_scaled = scaler.transform(test_set.reshape(1,-1)) ## One line of values, normalized.  These are the test values
# predictions = model.predict(X_test_scaled) ## Output - what should these values give you?

# print("Predicted: {}, actual: {}".format(predictions[0], 'M'));
