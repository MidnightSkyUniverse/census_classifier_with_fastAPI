"""
"""
import logging
import os
import yaml
import joblib

import pandas as pd
import numpy as np

from functions import save_model, data_encoder

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

    categorical_features = yaml.safe_load(open("params.yaml"))["cat_features"]
    label = yaml.safe_load(open("params.yaml"))["label"]
    training = yaml.safe_load(open("params.yaml"))["train"]["true_"]
    
    X, y, encoder, lb = data_encoder(X, categorical_features, label=label,training=training,encoder=False,lb=False)

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


    artifact_models = yaml.safe_load(open("params.yaml"))["model"]

    if training is True:
        for model, k in zip([encoder,lb], [artifact_models['encoder_pth'],artifact_models['lb_pth']]):
            logger.info(f"Saving {k} model")
            output = f"{pwd}/{k}"
            try:
                save_model(model,k)
            except BaseException:
                logger.error("Failed to save encoder and lb")



if __name__ == '__main__':

    process_data()
