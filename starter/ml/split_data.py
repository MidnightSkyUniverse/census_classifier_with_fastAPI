"""
Author: Ali Binkowska
Date: Dec 2021

The
"""
import logging
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def go():

    # Get the clean data
    artifacts = yaml.safe_load(open("params.yaml"))["data"]
    pth = f"{artifacts['clean_data']}"
    logger.info(f"Importe data from {artifacts['clean_data']}")
    try:
        data = pd.read_csv(pth)
    except FileNotFoundError:
        logger.error("Failed to load the file")

    # Split the data into training+validation and test data for test
    modeling = yaml.safe_load(open("params.yaml"))["modeling"]
    logger.info(f"Split the data with test_size={modeling['test_size']}")
    trainval, test = train_test_split(
        data,
        test_size=modeling['test_size'],
        random_state=modeling['random_state'],
    )

    # Save trainval and test data as csv files
    for df, k in zip([trainval, test], [
                     artifacts['trainval_data'], artifacts['test_data']]):
        logger.info(f"Saving {k} dataset")
        try:
            df.to_csv(k, index=False)
        except BaseException:
            logger.error("Failed to save file {k}")


if __name__ == '__main__':

    go()
