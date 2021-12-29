"""
Author: Ali Binkowska
Date: Dec 2021

This app is used to run Random Forest Classifier on census data using FastApi and Heroku'
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
#import uvicorn
import logging
import os

from starter.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DVC on Heroku - required code
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -d s3-bucket s3://uda3/storage")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


# Home site with welcome message - GET request
@app.get("/", tags=["home"])
async def get_root() -> dict:
    """
    Home page, returned as GET request
    """
    return {
        "message": "Welcome to FastAPI interface to Rnadom Forest Classifier of census data"
    }


# Class definition of the data that will be provided as POST request
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')
    salary: Optional[str]


# POST request to /predict site. Used to validate model with sample census data
@app.post('/predict')
async def predict(input: CensusData):
    """
    POST request that will provide sample census data and expect a prediction 

    Output:
        0 or 1
    """
    # Load model, encoder and lb
    try:
        model = joblib.load("./model/model.pkl")
        encoder = joblib.load("./model/encoder.pkl")
        lb = joblib.load("./model/lb.pkl")
    except BaseException:
        logging.error('Failed to load model, encoder or lb')

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

    # Read data sent as POST
    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])
    logger.info(f"Input data: {input_df}")

    # Process the data
    X_train, _, _, _ = process_data(
                input_df, categorical_features=cat_features, \
                label='salary', training=False, encoder=encoder, lb=lb)

    preds = int(model.predict(X_train)[0])
    logger.info(f"Preds: {preds}")
    return {"result": preds}


#if __name__ == "__main__":
#    """
#    Uvicorn will launch FastAPI website on localhost, port 8000
#    """
#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
