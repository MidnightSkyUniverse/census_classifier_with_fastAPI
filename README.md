# Censuse data analyes 

## What the project contains

### Conda environment
This project was setup using miniconda. Environment is stored in `census_fastAPI.yml`

### Model documentation
REAMDME defines how to run the project. Model card tells more about the model used and metrics
that comes with the model.
```
README.md
model_card.md
screenshots
```

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

                                                     +---------------------------------+
                                                     | starter/data/clean_data.csv.dvc |
                                                     +---------------------------------+
                                                                       *
                                                                       *
                                                                       *
                                                       +-----------------------------+
                                                       | starter/dvc.yaml:split_data |*******
                                               ********+-----------------------------+       **************
                                      *********                       *                                    ***************
                            **********                               *                                                    ***************
                       *****                                         *                                                                   **************
+------------------------+                          +-------------------------------+                                                                  ********
| starter/dvc.yaml:kfold |                          | starter/dvc.yaml:process_data |                                                                         *
+------------------------+                          +-------------------------------+                                                                         *
                                                    *****                         *****                                                                       *
                                               *****                                   ****                                                                   *
                                            ***                                            *****                                                              *
                      +--------------------------------+                                        ***                                                       *****
                      | starter/dvc.yaml:train_predict |                                          *                                            ***********
                      +--------------------------------+*******                                   *                                   *********
                                                               ***********                        *                        ***********
                                                                          *********               *               *********
                                                                                   ******         *         ******
                                                                                  +--------------------------------+
                                                                                  | starter/dvc.yaml:slice_predict |
                                                                                  +--------------------------------+


The original dataset stored in `data/census.csv` is cleaned with jupyter-notebook file EDA.jpynb.
All the other stages are executed with `dvc repro`
Firs stage is **split_data**. All the others depend on it.
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


#### Model metrics
All metrics are stored in `metrics` folder in json format

Use `dvc exp show` to show model performance. For short `dvc metrics show` will do.

#### Model tests
```
tests
sanitycheck.py
```
Tests are executed with `pytest` command as part of GitHu workflow. 
There are 7 tests implemented currently.
Script called `sanitycheck.py` is provided by Udacity and it's about to check
whether FastAPI is being properly tested with both POST and GET requests.

#### Heroku config files
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

#### Udacity
```
CODEOWNERS
LICENSE.txt
```




