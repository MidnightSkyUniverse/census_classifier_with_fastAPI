"""
Author: Ali Binkowska
Date: Dec 2021

The
"""
import os
import logging
import yaml
import json

# from numpy import array
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from model import  train_RandomForest_model, inference, compute_model_metrics, mean_calculation

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()


def go():

    trainval_pth = yaml.safe_load(open("params.yaml"))["data"]['trainval_data']
    X_pth = yaml.safe_load(open("params.yaml"))["data"]['X']
    y_pth = yaml.safe_load(open("params.yaml"))["data"]['y']
    pwd = os.getcwd()
    pth =  f"{pwd}/{trainval_pth}"
    logger.info(f"Importe data: trainval, X and y")
    try:
        trainval = pd.read_csv(trainval_pth)
        X = np.load(X_pth)
        y = np.load(y_pth)
    except FileNotFoundError:
        logger.error("Failed to load the files")

    logger.info(f"Initiate kFold")
    model_params =  yaml.safe_load(open("params.yaml"))["modeling"]
    kfold = KFold(
                  n_splits=model_params['n_splits'],
                  shuffle=model_params['shuffle'],
                  random_state=model_params['random_state']
    )

    # enumerate through splits
    metrics = list()
    logger.info('Enumerating through k-folds')

    for train_x_split, val_x_split in kfold.split(trainval):
        X_train, X_val = X[train_x_split], X[val_x_split]
        y_train, y_val = y[train_x_split], y[val_x_split]

        # train model
        model = train_RandomForest_model(X_train, y_train, model_params['random_state'])

        # inference
        preds = inference(model, X_val)


        precision, recall, fbeta = compute_model_metrics(y_val, preds)
        metrics.append((precision, recall, fbeta))

    # Present mean values for kfold runs
    kfold_pth = yaml.safe_load(open("params.yaml"))["metrics"]['kfold']
    logger.info(
        'Model RandomForest mean values for KFold-s saved to {kfold_pth}')
    precision_mean, recall_mean, fbeta_mean = mean_calculation(metrics)

    with open(kfold_pth, "w") as fd:
        json.dump(
            {"precision": precision_mean, "recall": recall_mean, "fbeta": fbeta_mean},
            fd,
            indent=4,
    )


if __name__ == '__main__':

    go()

