import pytest
import pandas as pd
import joblib
import sys
import os
#import starter.ml.data as dt

sys.path.insert(0, os.getcwd())
from starter.ml.functions import data_encoder


@pytest.fixture(scope='session')
def data_sample(request):
    payload = {
        'age': 31,
        'workclass': 'Private',
        'fnlgt': 45781,
        'education': 'Masters',
        'education-num': 14,
        'marital-status': 'Never-married',
        'occupation': 'Prof-speciality',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 14000,
        'capital-loss': 0,
        'hours-per-week': 55,
        'native-country': 'United-States',
        'salary': '>50K'
    }

    df = pd.DataFrame(payload, index=[0])
    return df


@pytest.fixture(scope='session')
def data(request):
    data = pd.read_csv('starter/data/clean_data.csv')
    return data


@pytest.fixture(scope='session')
def model(request):
    model = joblib.load("starter/model/model.pkl")
    return model


@pytest.fixture(scope='session')
def lb(request):
    lb = joblib.load("starter/model/lb.pkl")
    return lb


@pytest.fixture(scope='session')
def encoder(request):
    encoder = joblib.load("starter/model/encoder.pkl")
    return encoder


@pytest.fixture(scope='session')
def process_data_fixture(data, encoder, lb):
    '''
    Test data processing
    '''
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X = data.drop(['salary'], axis=1)

    X, y, _, _ = data_encoder(X, categorical_features=cat_features,
                              label=None, training=False, encoder=encoder, lb=lb)
    return X, y


@pytest.fixture(scope='session')
def process_sample_fixture(data_sample, encoder, lb):
    '''
    Test data sample processing
    '''

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X = data_sample.drop(['salary'], axis=1)

    X, y, _, _ = data_encoder(X, categorical_features=cat_features,
                              label=None, training=False, encoder=encoder, lb=lb)
    return X, y


@pytest.fixture(scope='session')
def json_sample(request):
    payload = {
        'age': 31,
        'workclass': 'Private',
        'fnlgt': 45781,
        'education': 'Masters',
        'education-num': 14,
        'marital-status': 'Never-married',
        'occupation': 'Prof-speciality',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 14000,
        'capital-loss': 0,
        'hours-per-week': 55,
        'native-country': 'United-States',
        'salary': '>50K'
    }

    return payload


@pytest.fixture(scope='session')
def json_sample_2(request):
    payload = {
        'age': 66,
        'workclass': 'Private',
        'fnlgt': 211781,
        'education': 'Masters',
        'education-num': 14,
        'marital-status': 'Never-married',
        'occupation': 'Prof-speciality',
        'relationship': 'Not-in-family',
        'race': 'Black',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 55,
        'native-country': 'United-States',
        'salary': '<=50K'
    }

    return payload


@pytest.fixture(scope='session')
def json_sample_with_error(request):
    payload = {
        'age': 66,
        'workclass': 'Private',
        'fnlgt': 211781,
        'education': "Masters",
        'education-num': "Fourteen", #14,
        'marital-status': 'Never-married',
        'occupation': 'Prof-speciality',
        'relationship': 'Not-in-family',
        'race': 'Black',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 55,
        'native-country': 'United-States',
        'salary': '<=50K'
    }

    return payload
