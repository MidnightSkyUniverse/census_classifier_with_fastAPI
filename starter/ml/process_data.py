"""
"""
import logging
import os
import yaml
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from model import save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()


def process_data():
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    trainval_pth = yaml.safe_load(open("params.yaml"))["data"]['trainval_data']
    logger.info(f"Import data from {trainval_pth}")
    pwd = os.getcwd()    
    try:
        X = pd.read_csv(f"{pwd}/{trainval_pth}")
    except FileNotFoundError:
        logger.error("Failed to load the file")

    logger.info("Split data into X and y") 
    label = yaml.safe_load(open("params.yaml"))["label"]
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # Separate categorical and numerical data  
    logger.info("Split X into categorical and continuous") 
    categorical_features = yaml.safe_load(open("params.yaml"))["cat_features"]
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    # Transofrm data  
    training = yaml.safe_load(open("params.yaml"))["train"]["true_"]
    if training is True:
        logger.info("Create OneHotEncoder and LabelBinarizer & transform data") 
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        # Return the flattened underlying data as an ndarray.
        y = lb.fit_transform(y.values).ravel()
    else:
        logger.info("Transform data with provided OneHotEncoder and LabelBinarizer") 
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    logger.info("Combine categorical and numerical data in X") 
    X = np.concatenate([X_continuous, X_categorical], axis=1)


    # Save the output of the function
    artifacts = yaml.safe_load(open("params.yaml"))["data"]

    for arr, k in zip([X, y], [artifacts['X'], artifacts['y']]):
        logger.info(f"Saving {k} array")
        output = f"{pwd}/{k}"
        try:
            #pd.DataFrame(df).to_csv(output, index=False)
            np.save(output,arr)
        except BaseException:
            logger.error(f"Failed to save file {k}")


    if training is True:
        for model, k in zip([encoder,lb], [artifacts['encoder_pth'],artifacts['lb_pth']]):
            logger.info(f"Saving {k} model")
            output = f"{pwd}/{k}"
            try:
                save_model(model,k)
            except BaseException:
                logger.error("Failed to save encoder and lb")



if __name__ == '__main__':

    process_data()
