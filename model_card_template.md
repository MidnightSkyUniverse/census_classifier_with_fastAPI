# Model Card



    Model Details such as who made it, type of model, training/hyperparameter details, and links to any additional documentation like a paper reference.
    Intended use for the model and the intended users.
    Metrics of how the model performs. Include overall performance and also key slices. A figure or two can convey a lot.
    Data including the training and validation data. How it was acquired and processed.
    Bias inherent either in data or model. This could also be included in the metrics or data section.
    Caveats, if there are any.


## Model Details
Author: Ali Binkowska
Data: Dec 2021

Model used is Random Forest Classifier. Dataset is census data with ~60,000 records.
The model predics whether a person is a group of people earning above or below $50K

The model is trained on 70% of dataset, 15% is reserved for validation and 15% for tests.

There is also an option to trin model on KFolds and separate metrics are generated for that


There are also predictions done on categorical slices of data. The results are stored to:
metrics/slice_scores.json


## Intended Use
That is a trianing project for Udacity nanodegree program.
The intention is to automate CI/CD with FastAPI and Heroku.
I use DVC to record pipeline stages and metrics. DVC with S3 is used to store artifacts remotely. 


## Training Data
Census data are used to train the model and predict whether a sample person
is in a group of those earning above or below $50,000


## Evaluation Data
15% of the census data is used to validate the model


## Metrics


## Ethical Considerations


## Caveats and Recommendations
