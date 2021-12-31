"""
Author: Ali Binkowska
Date: Dec 2021

The
"""
# from sklearn.model_selection import train_test_split
import logging
import yaml

# from numpy import array
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
# import joblib

from ml.data import process_data
from ml.model import train_RandomForest_model, compute_model_metrics, inference, \
    roc_curve_plot, mean_calculation, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)- %(message)s")
logger = logging.getLogger()


def go():

    trainval_pth = yaml.safe_load(open("params.yaml"))["data"]['trainval_data']
    pwd = os.getcwd()
    pth =  f"{pwd}/{trainval_pth}"
    logger.info(f"Importe data from {trainval_pth}")
    try:
        trainval = pd.read_csv(pth)
    except FileNotFoundError:
        logger.error("Failed to load the file")

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
        model = RandomForestClassifier(random_state=model_params['random_state'])
        model.fit(X_train, y_train)

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
        {
            "mean": 
                {"precision": precision_mean, "recall": recall_mean, "fbeta": fbeta_mean}
            
        },
        fd,
        indent=4,
    )


if __name__ == '__main__':

    process_data()

