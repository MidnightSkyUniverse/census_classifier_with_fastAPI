# Script to train machine learning model.

# from sklearn.model_selection import train_test_split
import hydra
import logging
from omegaconf import DictConfig

# from numpy import array
import pandas as pd
from sklearn.model_selection import KFold
# import joblib

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path='.', config_name='config')
def go(cfg: DictConfig):

    # Get the cleaned data
    logger.info(f"importing data from {cfg.data.cleaned_data}")
    pth = f"{hydra.utils.get_original_cwd()}/{cfg.data.dir}{cfg.data.cleaned_data}"
    data = pd.read_csv(pth)

    # prepare cross validation
    logger.info('Prepare cross validation with Kfold')
    kfold = KFold(n_splits=cfg['modeling']['n_splits'],\
                  shuffle=cfg['modeling']['shuffle'],\
                  random_state=cfg['modeling']['random_state'])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    logger.info('Split the data into X_train and y_train')
    # Separate X and y
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    # enumerate through splits
    # scores = list()
    logger.info('Enumerating through k-folds')
    for train_x_split, test_x_split in kfold.split(data):
        X_train, X_test = X[train_x_split], X[test_x_split]
        y_train, y_test = y[train_x_split], y[test_x_split]

        # train model
        logger.info('train the model')
        model = train_model(X_train, y_train)

        # inference
        logger.info('inference')
        preds = inference(model, X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        logger.info(f'precision: {precision}')
        logger.info(f'recall: {recall}')
        logger.info(f'fbeta: {fbeta}')


if __name__ == "__main__":
    go()
