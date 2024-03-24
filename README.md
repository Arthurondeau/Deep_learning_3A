[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.11+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![Code QC](https://github.com/Sanofi-GitHub/aida-imaging-graphanalysis/actions/workflows/on-push.yaml/badge.svg)](https://github.com/Sanofi-GitHub/aida-imaging-graphanalysis/actions/workflows/on-push.yaml)
[![coverage report](https://magellan-git.sanofi.com/root/ifgraphanalysis/badges/master/coverage.svg)](https://magellan-git.sanofi.com/root/ifgraphanalysis/commits/master)
# SETUP
- https://magellan.sanofi.com/pipeline/#/settings/profile 
  - Add "DATABRICKS_TOKEN" attribute with token generated in https://sanofi-rwe-emea.cloud.databricks.com/#mlflow/experiments
  - Add "DATABRICKS_HOST" attribute with value https://sanofi-rwe-emea.cloud.databricks.com/#mlflow/experiments

- source conda_install.sh
- source setup.sh

Install Python VSCode extension
Recommended vscode extensions: AutoDocstring, Sonarlint, GitLens

Set vscode python interpreter: CTRL+Maj+P -> Python: Select interpreter -> /root/anaconda3/envs/ifgraphs

# RUNNING

> :warning: :warning: :warning: The data paths in the config are absolute and assume that you are working from Magellan SCRIPT_DIR
> This can be changed in the conf/defaults.yaml


## Define a proper config for the experiment you want to run, taking the CD8_PD1_PDL1 template as model
Hydra config doc can be found here for additional details https://hydra.cc/docs/intro/
- folder names for local data storage
- `paths`:
    - `patientGraphs_name`
    - `subgraphs_name`
- `graph_creation.cell_filtering` for both IMMUCAN and MOSCATO, which will define the cell populations taken into account for the study (e.g. CD8, PD1, and PDL1 in the CD8_PD1_PDL1 config)
    - `detection_probability_filter`
    - `features_to_keep`
    - `features_to_filter`
    - `features_to_normalize`
- `dgi` `features_list` for both IMMUCAN and MOSCATO

## Run data processing steps to generate embeddings
- ```python scripts/IMMUCAN/reformat_workflow_format.py``` will transform data from workflow sample and IMC1/2 to the right format
>  :warning: :warning: :warning: Developped in a rush, paths are hardcoded which is bad but I haven't had time yet to improve it

Follow https://sanofi.atlassian.net/wiki/spaces/AIMOSC/pages/63835048885/Embeddings+generation+steps for the correct dataflow to generate embeddings
- ```python scripts/data_processing_pipeline.py``` will run all the steps in the above flow and copy data on Magellan s3 after each steps
- ```conf/curent_experiment.yaml``` has a ```hydra.sweeper.params``` section defining grid search-like set of parameters to run the pipeline with. add ```--multirun``` to trigger runs of the pipeline with all combinations of those parameters
- add ```env=training``` to specify the env  

**Running dgi_training.py will log results in Databricks to EXPERIMENTNAME_debugging whereas the env=training option would log it to EXPERIMENTNAME_training**  
This enables to disregard dev runs from actual training runs
## Transfer data to remote storage
By default, the code will store generated data locally, in the `data/`folder. Use pipe to copy it to the Bioimaging s3 bucket to save the results and share (the embeddings for instance)  

- Running ```python scripts/copy_data.py``` will copy data for the current experiment from local to remote  

```--target``` and ```--source``` can be added to copy specific folders. They should be in the format ```sequence,of,cfg["paths"],keys``` 

>  :warning: :warning: :warning: This script will override the target folder so be careful when using it. 

## Run dataset-specific analysis
scrips/notebooks has some visualization notebooks for data viz, exploration etc
### Immucan
#### TLS Analysis
  - sripts/IMMUCAN/tls_analysis
    - tls_analysis_pipeline.py
    - TLS_clustering.ipynb to viz results
