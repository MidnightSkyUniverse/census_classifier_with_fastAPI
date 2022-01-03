## Censuse data analyes 

This project has been written for Udaicty degree in DevOps.
It uses Random Forest Classifier to train and predict whether a sample person
will earn above or below  $50K per year based on cencus data used to train the model.

![model experiment with metrics](/screenshots/dvc_exp_show.png)

### Built With
Technologies used in the project
* [dvc pipeline](https://dvc.org/doc/start/data-pipelines)
* [GitHub](github.com) 
* [AWS S3](https://aws.amazon.com/s3/)
* [FastAPI](https://fastapi.tiangolo.com/) 
* [Heroku](https://heroku.com/)


### Getting Started

#### Conda environment
This project was setup using miniconda. To setup the enviornment run:
```conda env create -f census_fastAPI.yml```

#### Model documentation
This README defines how to run the project. [Model Card](model_card.md) tells more about 
the model used and metrics that comes with the model.
[Screenshots](/screenshots/) contains visuals from running the project from command line


#### Model execution
```
starter
  EDA.ipynb
  data/
  model/
  metrics/
  params.yaml
  dvc.yaml
  ml/
```
This project is built with `dvc pipeline`. Stages are defined in `dvc.yaml`. 
To visualise stages use `dvc dag`:
```
               +-------------------------+
               | data/clean_data.csv.dvc |
               +-------------------------+
                            *
                            *
                            *
                     +------------+
                     | split_data |***
                   **+------------+   ******
              *****        *                ******
           ***            *                       ******
        ***               *                             ******
+-------+         +--------------+                            ****
| kfold |         | process_data |                               *
+-------+         +--------------+                               *
                   **           **                               *
                 **               **                             *
               **                   **                           *
    +---------------+                 **                       ***
    | train_predict |                  *                   ****
    +---------------+***               *               ****
                        *****          *          *****
                             ****      *      ****
                                 ***   *   ***
                              +---------------+
                              | slice_predict |
                              +---------------+

```

The original dataset stored under `data/census.csv` is cleaned with jupyter-notebook `EDA.jpynb`.
All the other stages are executed with `dvc repro`.

#### Stages for dvc pipeline
```
> dvc stage list
split_data     Outputs data/test_data.csv, data/trainval_data.csv
process_data   Outputs data/X.npy, data/y.npy, model/encoder.pkl, model/lb.pkl
kfold          Reports metrics/kfold_scores.json
train_predict  Outputs model/model.pkl; Reports metrics/model_scores.json, metrics/precision.j…
slice_predict  Reports metrics/slice_scores.json
```

First stage is **split_data**. All the others depend on it.
Stage called **kfold** can be executed independently as it stores only metrics
Stages **process_data** has to predecess **train_predict**. 
Stage **slice_predict** depends on the model that is stored in stage **train_predict**.

Artifacts are stored by dvc to s3 remote storage. During CI phase executed by GitHub 
and CD phase executed by Heroku, the artifacts are pulled from storage.

Currently `dvc repro` does take all configuration from params.yaml.
In next version, there will be a possibility to overwrite arguments from command line.

##### About `ml/' folder
Script functions.py have functions that are used in pipeline stages.
All the other scripts in this folder represent pipeline stages.


### Model metrics
All metrics are stored in `metrics` folder in json format
Use `dvc exp show` to show model performance. For short `dvc metrics show` will do.
More about the metrics you can find in Model Card

### Model tests
```
tests
  conftest.py
  test_fastapi.py
  test_model.py
  sanitycheck.py
```
Tests are executed with `pytest` command as part of GitHu workflow. 
There are 7 tests implemented currently and the config is stored in `conftest.py`.
Script called `sanitycheck.py` is provided by Udacity and it's used to check
whether FastAPI is being properly tested with both POST and GET requests.
It can be run manually with `python` command.

### Heroku config files
```
Aptfile
Procfile
requirements.txt
runtime.txt
```
Continuouse Delivery is being done on Heroku with FastAPI providing web interface.

#### FastAPI
```
main.py
fastAPI_request_scripts
```
File main.py is defyining our web interface with FastAPI. Python scrips can be used to send
POST/GET requests to FastAPI running on Heroku or locally.


### GitHub workflow config
```
environment.yml
```
Continuous Integration is provided by GitHub.
`pylint` and `pytest` are executed and so before the app is developed to Heroku,
the checks have to pass on GitHub.


### Contact
Project Link: https://github.com/MidnightSkyUniverse/census_classifier_with_fastAPI



