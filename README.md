# gweb

This repository contains the project work carried out by group 97 as the final project of the MLOps course taught at DTU ([course website](https://skaftenicki.github.io/dtu_mlops/)). Group 97 consists of: Emilie Wenner, Anton Grier, Johanne Birk and Alberte Estad (see contributors list for individual github pages). 


1. **Overall goal:**

This project aims to detect Fraud in real financial transaction data based on IMB transactions. The main goal is to classify each account in the transactional list as has commited fraud or has not commited fraud.

1. **Framework:**
The [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#) ecosystem was used for this project. 
It provides specialized neural network layers designed for graphs and defines convolutional operations used in Graph neural networks.
GNN models use the local neighborhoods of nodes to learn embedded representations that capture the graph structure and
node feature

1. **Data:**

The [IBM dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml?fbclid=IwZXh0bgNhZW0CMTEAAR3zQXvP50SWwaXncNG7X-ot0RS16Ec8Yhg5vC3HkbB-t_WmdCyS7Ohb4vU_aem_GwSbK4o4M9XtLK0-ysi8Hg&select=HI-Medium_Trans.csv)
consists of a csv file containing 32 million transactions (edges) and 2 million accounts (nodes). Additionally a txt file contains information about all laundering attempts  (2756 attempts), where each laundring attempt constists of multiple transactions between multiple accounts, in total 35 thousand laundering transactions. This means that there is a strong class imbalance, which will be considered through regularization. We will load the data using PyTorch-Geometric interface.



1. **Deep learning models used?**
This project implements a  Graph   Neural   Network   model  and   incorporates different Graph Convolutional Network (GCN) layers, implemented in Pytorch Geometric.






## Project structure



The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
