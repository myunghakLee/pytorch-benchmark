#!/bin/bash
python run.py resnet50 -d cuda -m eager -t eval --profile
python run.py BERT_pytorch -d cuda -m eager -t eval --profile
python run.py dlrm -d cuda -m eager -t eval --profile
python run.py nvidia_deeprecommender -d cuda -m eager -t eval --profile