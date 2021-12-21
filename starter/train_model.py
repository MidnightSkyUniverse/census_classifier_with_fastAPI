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

    pwd = hydra.utils.get_original_cwd()

    # Get the clean data
    logger.info(f"importing data from {cfg.data.clean_data}")
    pth = f"{hydra.utils.get_original_cwd()}/{cfg.data.clean_data}"
    try:
        data = pd.read_csv(pth)
    except FileNotFoundError:
        logger.error(f"Failed to load the file")

    # Split the data into training+validation and test data for test
    trainval, test = train_test_split(data, \
                                    test_size=cfg.modeling.test_size,\
                                     random_state=cfg.modeling.random_state)
    
    # Save test data in separate file
    logger.info(f"Save the data for test to {cfg.data.test_data}")
    pth = f"{pwd}/{cfg.data.test_data}"
    try:
        test.to_csv(pth)
    except:
        logger.error(f"Failed to save test file") 

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
    # Separate X and y
    logger.info('Encode the data')
    X, y, encoder, lb = process_data(
        trainval, categorical_features=cat_features, label="salary", training=True
    )
    
    logger.info(f"Save encoder and lb")
    try:
        pth = f"{pwd}/{cfg.data.encoder_pth}"
        #save_model(encoder,pth)
        pth = f"{pwd}/{cfg.data.lb_pth}"
        #save_model(lb,pth)
    except:
        logger.error(f"Failed to save encoder and lb") 
    

    
    # Prepare cross validation
    kfold = KFold(n_splits=cfg.modeling.n_splits,
                  shuffle=cfg.modeling.shuffle,
                  random_state=cfg.modeling.random_state)

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
    logger.info(f"Save ROC curve to {cfg.metrics.dir}{cfg.metrics.roc}")
    pth = f"{pwd}/{cfg.metrics.dir}{cfg.metrics.roc}" 
    roc_curve_plot(model, y_val, preds,pth)

    # Check model performance on slices, record the performance
    logger.info('Model performance on slices')
    with open(f"{pwd}/{cfg.data.slice_performance}", "w") as file1:
        for cat in cat_features:
            unique_values = trainval[cat].unique()
            for value in  unique_values:
                X, y, encoder, lb = process_data(
                    trainval[trainval[cat]==value], categorical_features=cat_features, \
                    label="salary", training=False, encoder=encoder, lb=lb
                )
                preds = inference(model, X)
                precision, recall, fbeta = compute_model_metrics(y, preds)
                file1.write(f"Category: {cat}, Value: {value}\n")
                file1.write(f"Metrics:\n")
                file1.write(f"Precision: {precision}\n")
                file1.write(f"Recall: {recall}\n")
                file1.write(f"Fbeta: {fbeta}\n\n")
    logger.info(f"Model performance recorded to {cfg.data.slice_performance}")

if __name__ == "__main__":
    go()
