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
from ml.model import train_KNeighbours_model, train_LogisticRegression_model, \
        train_RandomForest_model, compute_model_metrics, inference, roc_curve_plot

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
        logger.info('train the model')
        model_1 = train_KNeighbours_model(X_train, y_train)
        model_2 = train_LogisticRegression_model(X_train, y_train, cfg.modeling.random_state, cfg.modeling.max_iter)
        model_3 = train_RandomForest_model(X_train, y_train,cfg.modeling.random_state)

        # inference
        logger.info('inference')
        preds_1 = inference(model_1, X_test)
        preds_2 = inference(model_2, X_test)
        preds_3 = inference(model_3, X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, preds_1)
        metrics_1.append((model_1, precision, recall, fbeta))

        precision, recall, fbeta = compute_model_metrics(y_test, preds_2)
        metrics_2.append((model_2, precision, recall, fbeta))

        precision, recall, fbeta = compute_model_metrics(y_test, preds_3)
        metrics_3.append((model_3, precision, recall, fbeta))


    # Select model with highest precision and save it
    precision_ = 0
    for model, precision, recall, fbeta in metrics_1:
    #    print(f".....precision: {precision}, recall: {recall}, fbeta: {fbeta}")

        if precision > precision_:
            model_1 = model
            precision_ = precision
            recall_ = recall
            fbeta = fbeta

    print(f"best model_1 precision: {precision}")

    precision_ = 0
    for model, precision, recall, fbeta in metrics_2:
     #   print(f".....precision: {precision}, recall: {recall}, fbeta: {fbeta}")

        if precision > precision_:
            model_2 = model
            precision_ = precision
            recall_ = recall
            fbeta = fbeta

    print(f"best model_2 precision: {precision}")

    precision_ = 0
    for model, precision, recall, fbeta in metrics_3:
      #  print(f".....precision: {precision}, recall: {recall}, fbeta: {fbeta}")

        if precision > precision_:
            model_3 = model
            precision_ = precision
            recall_ = recall
            fbeta_ = fbeta

    print(f"best model_3 precision: {precision}")

    # Select best performing model and draw roc_curve
    pth = f"{hydra.utils.get_original_cwd()}/{cfg.metrics.dir}{cfg.metrics.roc}" 
    roc_curve_plot(model_1,model_2,model_3,X_test,y_test,pth)


if __name__ == "__main__":
    go()
