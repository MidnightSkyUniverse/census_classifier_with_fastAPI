# Script to train machine learning model.

# from sklearn.model_selection import train_test_split
import hydra
import logging
from omegaconf import DictConfig

# from numpy import array
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
# import joblib

from ml.data import process_data
from ml.model import train_KNeighbours_model, train_LogisticRegression_model, \
        train_RandomForest_model, compute_model_metrics, inference, roc_curve_plot, \
        mean_calculation

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
    kfold = KFold(n_splits=cfg['modeling']['n_splits'],
                  shuffle=cfg['modeling']['shuffle'],
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
    metrics_1 = list()
    metrics_2 = list() 
    metrics_3 = list()
    logger.info('Enumerating through k-folds')
    for train_x_split, test_x_split in kfold.split(data):
        X_train, X_test = X[train_x_split], X[test_x_split]
        y_train, y_test = y[train_x_split], y[test_x_split]

        # train model
        #logger.info('train the model')
        model_1 = train_KNeighbours_model(X_train, y_train)
        model_2 = train_LogisticRegression_model(X_train, y_train, cfg.modeling.random_state, cfg.modeling.max_iter)
        model_3 = train_RandomForest_model(X_train, y_train,cfg.modeling.random_state)

        # inference
        #logger.info('inference')
        preds_1 = inference(model_1, X_test)
        preds_2 = inference(model_2, X_test)
        preds_3 = inference(model_3, X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, preds_1)
        metrics_1.append((precision, recall, fbeta))

        precision, recall, fbeta = compute_model_metrics(y_test, preds_2)
        metrics_2.append((precision, recall, fbeta))

        precision, recall, fbeta = compute_model_metrics(y_test, preds_3)
        metrics_3.append((precision, recall, fbeta))


    # Select model with highest precision and save it
    logger.info('Model KNeighbours mean values')
    precision_mean, recall_mean, fbeta_mean = mean_calculation(metrics_1)
    logger.info(f"Precision: {precision_mean}'; Recall: {recall_mean}; Fbeta: {fbeta_mean}")

    logger.info('Model LogisticRegression mean values')
    precision_mean, recall_mean, fbeta_mean = mean_calculation(metrics_2)
    logger.info(f"Precision: {precision_mean}'; Recall: {recall_mean}; Fbeta: {fbeta_mean}")

    logger.info('Model RandomForestmean values')
    precision_mean, recall_mean, fbeta_mean = mean_calculation(metrics_3)
    logger.info(f"Precision: {precision_mean}'; Recall: {recall_mean}; Fbeta: {fbeta_mean}")

    # Train two better performing models on full data set
    logger.info('Train LogisticRegression on full data set') 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = cfg.modeling.test_size, \
                                        random_state=cfg.modeling.random_state)
    model_LR = train_LogisticRegression_model(X_train, y_train, cfg.modeling.random_state, cfg.modeling.max_iter)
    model_RF = train_RandomForest_model(X_train, y_train,cfg.modeling.random_state)

    preds_LR = inference(model_LR, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds_LR)
    logger.info(f"LogisticRegression: precision: {precision}'; recall: {recall}; fbeta: {fbeta}")

    preds_RF = inference(model_RF, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds_RF)
    logger.info(f"RandomForest: precision: {precision}'; recall: {recall}; fbeta: {fbeta}")
    
    # Draw roc_curve
    pth = f"{hydra.utils.get_original_cwd()}/{cfg.metrics.dir}{cfg.metrics.roc}" 
    roc_curve_plot(model_LR, model_RF, y_test, preds_LR,preds_RF,pth)


if __name__ == "__main__":
    go()
