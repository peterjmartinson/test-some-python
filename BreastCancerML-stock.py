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

data = pd.read_csv('Data/data.csv', index_col=False)
print(data.head(5))

Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values

# Below shows SVC is better than Logistics model; DO NOT NEED FOR PRODUCTION!
num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=123)
start = time.time()
cv_results = cross_val_score(LogisticRegression(), X, Y, cv=kfold, scoring='accuracy')
end = time.time()
print( "Logistics regression accuracy: %f, run time: %f)" % (cv_results.mean(), end-start))

start = time.time()
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
cv_results = cross_val_score(SVC(C=2.0, kernel="rbf"), X_scaled, Y, cv=kfold, scoring='accuracy')
end = time.time()
print( "SVC accuracy: %f, run time: %f)" % (cv_results.mean(), end-start))

# THIS IS WHAT IS NEEDED FOR PRODUCTION TO ESTIMATE THE MODEL
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
model = SVC(C=2.0, kernel='rbf')
model.fit(X_scaled, Y)


# THIS IS WHAT IS NEEDED FOR PREDICTION
X_test_scaled = scaler.transform(X[10,:].reshape(1,-1))
predictions = model.predict(X_test_scaled)

print("Predicted: {}, actual: {}".format(predictions[0], Y[10]));
