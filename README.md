[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.11+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![Code QC](https://github.com/Sanofi-GitHub/aida-imaging-graphanalysis/actions/workflows/on-push.yaml/badge.svg)](https://github.com/Sanofi-GitHub/aida-imaging-graphanalysis/actions/workflows/on-push.yaml)
[![coverage report](https://magellan-git.sanofi.com/root/ifgraphanalysis/badges/master/coverage.svg)](https://magellan-git.sanofi.com/root/ifgraphanalysis/commits/master)
# SETUP

- source conda_install.sh
- source setup.sh

Install Python VSCode extension
Recommended vscode extensions: AutoDocstring, Sonarlint, GitLens

Set vscode python interpreter: CTRL+Maj+P -> Python: Select interpreter -> /root/anaconda3/envs/DL3A

# RUNNING

> :warning: :warning: :warning: The data paths in the config aren't absolute and assume that you keep the same folder arborescence
> This can be changed in the conf/defaults.yaml

## All parameters and model's parameters are defined in config file with hydra library (allow to automatically instantiate models classes)
Hydra config doc can be found here for additional details https://hydra.cc/docs/intro/

## Run data processing steps to generate embeddings
- ```python scripts/npz_to_pck.py``` will transform raw data to subsampled data with indicated channels (in conf/training/defaults.yaml)

## Define a proper training config
All training parameters can be accessed in the config file in conf/training/defaults.yaml 

## Models Zoo and how to choose one
The model type used for training can be accessed in defaults config file at training_args:model_type where *model_type* should correspond to a key in 
the config file in model_dict/defaults.yaml (where all models parameters are defined)

## RUN a training
**Running model_training.py will log results in tracking_uri folder based on mlflow workflow**  
The folder can be accessed in conf/training/defaults.yaml with the key "tracking_uri".

## Open run experiment with MLFLOW
To see run experiment logs run *mlflow ui --backend-store-uri tracking_uri_folder_name -p ####*
