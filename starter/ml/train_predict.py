"""
Author: Ali Binkowska
Date: Dec 2021

The
"""
# from sklearn.model_selection import train_test_split
import os
import logging
import yaml
import json

# from numpy import array
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split

from model import  train_RandomForest_model, inference, compute_model_metrics, mean_calculation, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()


def go():

    X_pth = yaml.safe_load(open("params.yaml"))["data"]['X']
    y_pth = yaml.safe_load(open("params.yaml"))["data"]['y']
    pwd = os.getcwd()
    logger.info(f"Importe data: trainval, X and y")
    try:
        X = np.load(X_pth)
        y = np.load(y_pth)
    except FileNotFoundError:
        logger.error("Failed to load the files")


     # Train the model on full data set
    logger.info('Train RandomForest on full data set')
    model_params =  yaml.safe_load(open("params.yaml"))["modeling"]

    X_train, X_val, y_train, y_val = train_test_split(
            X, 
            y, 
            test_size=model_params['test_size'], 
            random_state=model_params['random_state'],
    )
    model = train_RandomForest_model(
            X_train, 
            y_train, 
            model_params['random_state']
    )

    preds = inference(model, X_val)
    precision, recall, fbeta = compute_model_metrics(y_val, preds)
    logger.info(f"Full dataset run: precision: {precision}'; recall: {recall}; fbeta: {fbeta}")
    model_scores = yaml.safe_load(open("params.yaml"))["metrics"]['model']
    with open(model_scores, "w") as fd:
        json.dump(
            {"precision": precision, "recall": recall, "fbeta": fbeta},
            fd,
            indent=4,
        )

    # Save ROC 
    roc_pth = yaml.safe_load(open("params.yaml"))["metrics"]['roc']
    logger.info(f"Save ROC curve to {cfg.metrics.roc}")
    pth = f"{pwd}/{cfg.metrics.dir}{cfg.metrics.roc}"
    roc_curve_plot(model, y_val, preds, pth)


    # Save production model
    model_pth = yaml.safe_load(open("params.yaml"))["data"]['model_pth']
    output = f"{pwd}/{model_pth}"
    try:
        save_model(model,model_pth)
    except BaseException:
        logger.error("Failed to save model")


    
if __name__ == '__main__':

    go()

