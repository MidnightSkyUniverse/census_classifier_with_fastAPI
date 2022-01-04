# Model Card

- Author: Ali Binkowska
- Last updated: Jan 2022


## Model Details

Model used is Random Forest Classifier. Dataset is census data with ~60,000 records.
The model predics whether a sample person belongs to  a group of people earning above or below $50K.

The model is trained on 70% of dataset, 15% is reserved for validation and 15% for tests.

There isan option to trin model on KFolds and separate metrics are generated for that

There are also predictions done on categorical slices of data. The results are stored to:
`metrics/slice_scores.json`


## Intended Use
That is a trianing project for Udacity nano degree program.
The intention is to automate CI/CD for Machine Learning project with FastAPI and Heroku.
I use DVC to record pipeline stages and metrics. DVC with S3 is used to store artifacts remotely.


## Training Data
Census data are provided with the project under `data/census.csv`


## Evaluation Data
15% of the census data is used to validate the model


## Metrics

Metrics are stored in `starter/metrics/` folder.
For the model, average precision and ROC AUC is being calculated:

![sample metrics](/screenshots/dvc_exp_show.png)

Model parameters can be changed at `starter/params.yaml` file.


Kfolds metrics are stored in form of json file (sample below):
```json
            "precision": 0.7523427041499331,
            "recall": 0.5884816753926702,
            "fbeta": 0.6603995299647474

```

And slices have metrics stored for every feature:
 ```json
            "Category": "occupation",
            "Value": "Craft-repair",
            "precision": 0.9833333333333333,
            "Recal": 0.918562874251497,
            "FBeta": 0.94984520123839
```

## Ethical Considerations
Census data are data collected from people. The answers depend on how those people would like
themselves to be seen and represented. And depend on how much people want to reveal about themselves.
This kind od data can be influenced by politics, other people from the bubbles we live in
and our wishes.
As per slice performance metrics, some categorical values bring ideal predition.
The data are insufficient for 'nationality' feature to trust that the model would handle well
larger data set.


## Caveats and Recommendations
This model has very basic parameters set and before it is used to do real job, the parameters
scope should be extended. As this project was about CI/CD, more emphasis was put
on having the project utilizes tools like GitHub, dvc+S3 and Heroku than on how well
to configure the model
