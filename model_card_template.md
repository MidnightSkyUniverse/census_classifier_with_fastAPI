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
Model is set to perform 150 max_iter, the model is trained on 60% of dataset, 20% is reserved
for validation and 20% for tests.

Hydra is used to manage variables. Hydra config can be found under:
starter/config.yaml

There are also predictions done on categorical slices of data. The results are stored to:
metrics/slice_performance.txt

ROC curve chart is presented under:
metrics/roc_curve.png

## Intended Use
That is a trianing project for Udacity nanodegree program.
The intention is to automate CI/CD with FastAPI and Heroku. 


## Training Data
Census data are used to train the model and predict whether a sample person
is in a group of those earning above or below $50,000


## Evaluation Data
20% of the census data is used to validate the model


## Metrics


## Ethical Considerations


## Caveats and Recommendations
