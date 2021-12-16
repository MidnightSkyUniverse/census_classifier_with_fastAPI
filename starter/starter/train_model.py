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
from ml.model import train_RandomForest_model, compute_model_metrics, inference, \
            roc_curve_plot, mean_calculation, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path='.', config_name='config')
def go(cfg: DictConfig):

    # Get the clean data
    logger.info(f"importing data from {cfg.data.clean_data}")
    pth = f"{hydra.utils.get_original_cwd()}/{cfg.data.dir}{cfg.data.clean_data}"
    data = pd.read_csv(pth)

    # Split the data into training+validation and test data that will be saved for the test
    trainval, test = train_test_split(data, \
                                    test_size=cfg.modeling.test_size,\
                                     random_state=cfg.modeling.random_state)
    pth = f"{hydra.utils.get_original_cwd()}/{cfg.data.dir}{cfg.data.test_data}"
    test.to_csv(pth)

    # Data encoding 
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
    logger.info('Encode the data')
    # Separate X and y
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    # Prepare cross validation
    kfold = KFold(n_splits=cfg['modeling']['n_splits'],
                  shuffle=cfg['modeling']['shuffle'],
                  random_state=cfg['modeling']['random_state'])

    # enumerate through splits
    metrics = list()
    logger.info('Enumerating through k-folds')
    for train_x_split, val_x_split in kfold.split(trainval):
        X_train, X_val = X[train_x_split], X[val_x_split]
        y_train, y_val = y[train_x_split], y[val_x_split]

        # train model
        model = train_RandomForest_model(X_train, y_train,cfg.modeling.random_state)

        # inference
        preds = inference(model, X_val)

        precision, recall, fbeta = compute_model_metrics(y_val, preds)
        metrics.append((precision, recall, fbeta))


    # Present mean values for kfold runs
    logger.info('Model RandomForestmean mean values for {cfg.modeling.n_splits} runs')
    precision_mean, recall_mean, fbeta_mean = mean_calculation(metrics)
    logger.info(f"Precision: {precision_mean}'; Recall: {recall_mean}; Fbeta: {fbeta_mean}")

    # Train the model on full data set
    logger.info('Train RandomForest on full data set') 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = cfg.modeling.test_size, \
                                        random_state=cfg.modeling.random_state)
    model = train_RandomForest_model(X_train, y_train,cfg.modeling.random_state)
    preds = inference(model, X_val)
    precision, recall, fbeta = compute_model_metrics(y_val, preds)
    logger.info(f"Full dataset run: precision: {precision}'; recall: {recall}; fbeta: {fbeta}")

    # Save production model
    pth = f"{hydra.utils.get_original_cwd()}/{cfg.data.model_pth}"
    save_model(model,pth)
    
    # Draw roc_curve
    #pth = f"{hydra.utils.get_original_cwd()}/{cfg.metrics.dir}{cfg.metrics.roc}" 
    #roc_curve_plot(model, y_val, preds,pth)

    # Train model on slices


if __name__ == "__main__":
    go()
