"""
Author: Ali Binkowska
Date: Dec 2021

This file contains functions used in the pipeline to train and test model
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

import joblib
import numpy as np
from numpy import mean


def data_encoder(
        X,
        categorical_features,
        label=None,
        training=False,
        encoder=False,
        lb=False):

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    # Transofrm data
    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        # Return the flattened underlying data as an ndarray.
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, encoder, lb


def train_KNeighbours_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    return model


def train_LogisticRegression_model(X_train, y_train, random_state, max_iter):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    model.fit(X_train, y_train)

    return model


def train_RandomForest_model(X_train, y_train, random_state):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y_test, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y_test, preds, beta=1, zero_division=1)
    precision = precision_score(y_test, preds, zero_division=1)
    recall = recall_score(y_test, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X_test):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X_test)
    return preds

def mean_calculation(metrics):

    # Select model with highest precision and save it
    precision, recall, fbeta = zip(*metrics)
    return mean(precision), mean(recall), mean(fbeta)


def save_model(model, pth):
    '''
             saves model to ./models as .pkl file
                input:
                    model: trained model
                    pth: path to store the model
    '''
    joblib.dump(model, pth)


def load_model(pth):
    '''
             saves model to ./models as .pkl file
                input:
                    pth: path to store the model
    '''
    return joblib.load(pth)
