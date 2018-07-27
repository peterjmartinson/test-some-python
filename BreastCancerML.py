import numpy as np
import pandas as pd

import mysql.connector
from mysql.connector import errorcode
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

    model = SVC(C=2.0, kernel='rbf') ## choose the model
    scaler = None

    def __init__(self, csv):
        """
        Initialize the machine with canned training data
        """
        pandas_data = pd.read_csv(csv, index_col=False)
        Y = pandas_data['diagnosis'].values
        X = pandas_data.drop('diagnosis', axis=1).values
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, Y)

    # def trainFromCSV(self, csv):
    #     pandas_data = pd.read_csv(csv, index_col=False)
    #     Y = pandas_data['diagnosis'].values
    #     X = pandas_data.drop('diagnosis', axis=1).values
    #     self.scaler = StandardScaler().fit(X)
    #     X_scaled = self.scaler.transform(X)
    #     self.model.fit(X_scaled, Y)

    def trainFromDB(self, user, password, host, database):
        """
        Retrain the model using data from a database
        """
        cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        mycursor=cnx.cursor()
        query = ("SELECT diagnosis FROM UW_Data")
        mycursor.execute(query)
        diagnosis=[]
        for record in mycursor:
            for field in record:
                diagnosis.append(field)
        query = ('''
          SELECT
            id
          , radius_mean
          , texture_mean
          , perimeter_mean
          , area_mean
          , smoothness_mean
          , compactness_mean
          , concavity_mean
          , concave points_mean
          , symmetry_mean
          , fractal_dimension_mean
          , radius_se
          , texture_se
          , perimeter_se
          , area_se
          , smoothness_se
          , compactness_se
          , concavity_se
          , concave points_se
          , symmetry_se
          , fractal_dimension_se
          , radius_worst
          , texture_worst
          , perimeter_worst
          , area_worst
          , smoothness_worst
          , compactness_worst
          , concavity_worst
          , concave points_worst
          , symmetry_worst
          , fractal_dimension_worst
          FROM
            UW_Data
          ''')
        mycursor.execute(query)

        data=[]
        for record in mycursor:
            patient=[]
            for field in record:
                patient.append(field)
            data.append(patient)

        Y = diagnosis
        X = data
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, Y)

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
        if type(data) != np.ndarray:
            sample_data = self.setInput(data)
        else:
            sample_data = data
        sample_data_scaled = self.scaler.transform(sample_data.reshape(1,-1)) ## One line of values, normalized.  These are the test values
        predictions = self.model.predict(sample_data_scaled) ## Output - what should these values give you?
        return predictions[0]
  

if __name__ == '__main__':
    input_data = 'Data/data.csv'
    data = pd.read_csv(input_data, index_col=False)

    Y = data['diagnosis'].values # Get everything down the diagnosis column
    X = data.drop('diagnosis', axis=1).values # get everything *but* the diagnosis column

    machine = BreastCancerPredictionMachine('Data/data.csv')
    result_1 = machine.getDiagnosis(X[10,:])
    result_2 = machine.getDiagnosis(X[20,:])
    result_3 = machine.getDiagnosis([8510426,13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259])

    print("Predicted: {}, actual: {}".format(result_1, Y[10]));
    print("Predicted: {}, actual: {}".format(result_2, Y[20]));
    print("Predicted: {}, actual: {}".format(result_3, 'B'));

