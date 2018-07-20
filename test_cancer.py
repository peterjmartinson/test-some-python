from BreastCancerML import *

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





