from BreastCancerML import *

# import numpy as np
# import pandas as pd

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# import time

def test_read_csv():
    data = pd.read_csv('Data/data.csv', index_col=False)
    assert data.empty == False

def test_pumper_NoneInput():
    pumper = ModelPumperOuter()
    input_data = None
    result = pumper.getData(input_data)
    assert result == 0

def test_pumper_SomeInput():
    pumper = ModelPumperOuter()
    input_data = 'data.csv'
    result = pumper.getData(input_data)
    assert result == 1

def test_pumper_NotCsv():
    pumper = ModelPumperOuter()
    input_data = 'data'
    result = pumper.getData(input_data)
    assert result == 0







# v = UnixModemConfigurator()
# h = HayesModem()
# z = ZoomModem()
# e = ErnieModem()

# def test_HayesForUnix():
#   h.Accept(v);
#   assert  "&s1=4&D=3" == h.configurationString

# def test_ZoomForUnix():
#   z.Accept(v);
#   assert  42 == z.configurationValue

# def test_HayesForUnix():
#   e.Accept(v);
#   assert  "C is too slow" == e.internalPattern



