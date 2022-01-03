"""
Author: Ali Binkowska
Date: Dec 2021

This script trains and evaluates the Random Forest Classifier that uses parameters from params.yaml file
"""
# from sklearn.model_selection import train_test_split
import logging
import yaml
import json
import math

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

from functions import train_RandomForest_model, inference, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()


def go():

    X_pth = yaml.safe_load(open("params.yaml"))["data"]['X']
    y_pth = yaml.safe_load(open("params.yaml"))["data"]['y']
    logger.info("Importe data: trainval, X and y")
    try:
        X = np.load(X_pth)
        y = np.load(y_pth)
    except FileNotFoundError:
        logger.error("Failed to load the files")

     # Train the model on full data set
    logger.info('Train RandomForest on full data set')
    model_params = yaml.safe_load(open("params.yaml"))["modeling"]

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

    logger.info(
        "Calculate metrics: precision, recall, prc_thresholds, ROC curve")
    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        y_val, preds)
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_val, preds)

    avg_prec = metrics.average_precision_score(y_val, preds)
    roc_auc = metrics.roc_auc_score(y_val, preds)

    # Save average precision metrics for the model
    artifacts = yaml.safe_load(open("params.yaml"))["metrics"]
    logger.info(
        f"Average precision and ROC metrics saved to {artifacts['model']}")
    logger.info(
        f"Precision metrics - avg_prec:{avg_prec} and roc_auc:{roc_auc}")
    with open(artifacts['model'], "w") as fd:
        json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    logger.info(f"Precision metrics saved to {artifacts['precision']}")
    with open(artifacts['precision'], "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    # zip(precision, recall, prc_thresholds)
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
            default=np_encoder,
        )

    logger.info(f"ROC metrics saved to {artifacts['roc']}")
    with open(artifacts['roc'], "w") as fd:
        json.dump(
            {
                "roc": [
                    {"fpr": fp, "tpr": tp, "threshold": t}
                    for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                ]
            },
            fd,
            indent=4,
            default=np_encoder
        )

    # Save production model
    model_pth = yaml.safe_load(open("params.yaml"))["model"]['model_pth']
    logger.info(f"Model saved to {model_pth}")
    try:
        save_model(model, model_pth)
    except BaseException:
        logger.error("Failed to save model")


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


if __name__ == '__main__':

    go()
