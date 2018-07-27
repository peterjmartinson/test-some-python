"""
__author__ = peterjmartinson
"""
from BreastCancerML import BreastCancerPredictionMachine


# db = 'the database'
# username = 'username'
# password = 'password'
# url = 'http://location.of.the.database'
# machine = BreastCancerPredictionMachine(url, username, password, db)

training_data = 'Data/data.csv'
machine = BreastCancerPredictionMachine(training_data)

test_data = [8510426,13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]

result = machine.getDiagnosis(test_data)

print(result)
