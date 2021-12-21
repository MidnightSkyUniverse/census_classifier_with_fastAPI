import numpy as np



def test_data_size(data):
    """
    Test if we have reasonable amount of data
    """
    assert 1000 < data.shape[0] < 1000000


def test_process_data(process_data_fixture):
    """
    Test shape of data after encoding
    """
    assert process_data_fixture[0].shape[0] > 1000


def test_predict(model, process_data_fixture):
    '''
    Test model on sample data - expected prediction is equal 1
    '''
    assert isinstance(model.predict(process_data_fixture[0]), np.ndarray)


def test_predict_sample(model, process_sample_fixture):
    '''
    Test model on sample data - expected prediction is equal 1
    '''
    assert int(model.predict(process_sample_fixture[0])[0]) == 1
