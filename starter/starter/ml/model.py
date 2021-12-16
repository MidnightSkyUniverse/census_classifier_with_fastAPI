from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

import joblib
from numpy import mean



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

def roc_curve_plot(model, test_y, preds, pth):
    '''
        creates and stores the feature importances in pth
        input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
        output:
             None
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test_y, preds)
    #roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr,tpr,label="Random Forest")
    plt.savefig(pth, bbox_inches='tight')
    plt.clf()



def mean_calculation(metrics):

    # Select model with highest precision and save it
    precision, recall, fbeta = zip (*metrics)
    return mean(precision), mean(recall), mean(fbeta)

   
def save_model(model,pth):
    '''
             saves model to ./models as .pkl file
                input:
                    model: trained model
                    pth: path to store the model
    '''
    joblib.dump(model, pth)
