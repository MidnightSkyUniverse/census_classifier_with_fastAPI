"""
Author: Ali Binkowska
Date: Dec 2021

This script uses saved model, encoder and lb to evaluate model performance on slices of categorical data
"""
# from sklearn.model_selection import train_test_split
import os
import logging
import yaml
import json

import pandas as pd
import numpy as np

from functions import compute_model_metrics, load_model, data_encoder, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()


def go():

    slice_scores = yaml.safe_load(open("params.yaml"))[
        "metrics"]['slice_scores']
    logger.info(f'Test model performance on slices and save to {slice_scores}')

    logger.info("Load trainval data set")
    trainval_pth = yaml.safe_load(open("params.yaml"))["data"]['trainval_data']
    pwd = os.getcwd()
    try:
        trainval = pd.read_csv(f"{pwd}/{trainval_pth}")
    except FileNotFoundError:
        logger.error("Failed to load the dataset")

    artifacts = yaml.safe_load(open("params.yaml"))["model"]
    try:
        model = load_model(f"{pwd}/{artifacts['model_pth']}")
        encoder = load_model(f"{pwd}/{artifacts['encoder_pth']}")
        lb = load_model(f"{pwd}/{artifacts['lb_pth']}")
    except FileNotFoundError:
        logger.error("Failed to load the dataset")

    cat_features = yaml.safe_load(open("params.yaml"))["cat_features"]
    label = yaml.safe_load(open("params.yaml"))["label"]
    training = yaml.safe_load(open("params.yaml"))["train"]['false_']
    metrics = []

    for cat in cat_features:
        unique_values = trainval[cat].unique()
        for value in unique_values:
            X, y, encoder, lb = data_encoder(
                trainval[trainval[cat] == value], categorical_features=cat_features,
                label=label, training=training, encoder=encoder, lb=lb
            )
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            metrics.append((cat, value, precision, recall, fbeta))

    slice_scores = yaml.safe_load(open("params.yaml"))[
        "metrics"]['slice_scores']
    logger.info(f"Slice scores saved to {slice_scores}")
    with open(f"{pwd}/{slice_scores}", "w") as fd:
        json.dump(
            {
                "Slice Scores": [
                    {"Category": c, "Value": v, "precision": p, "Recal": r, "FBeta": f}
                    for c, v, p, r, f in metrics
                ]
            },
            fd,
            indent=4,
            default=np_encoder
        )


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


if __name__ == '__main__':

    go()
