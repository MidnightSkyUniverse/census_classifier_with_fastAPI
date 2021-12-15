from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

import joblib




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

def train_LogisticRegression_model(X_train, y_train,random_state,max_iter):
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


def train_RandomForest_model(X_train, y_train,random_state):
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

def roc_curve_plot(model_1, model_2, model_3,X_test,y_test,pth):
    '''
        creates and stores the feature importances in pth
        input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
        output:
             None
    '''
    ax = plt.gca()
    fig = plot_roc_curve(model_1, X_test, y_test, ax=ax, alpha=0.8)
    fig = plot_roc_curve(model_2, X_test, y_test, ax=ax, alpha=0.8)
    fig = plot_roc_curve(model_3, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(pth, bbox_inches='tight')
    plt.clf()
    #lrc_plot = plot_roc_curve(model_2, X_test, y_test, ax=ax, alpha=0.8)
