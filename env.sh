#!/bin/bash

echo "###### Step1 ######\n"
git clone https://github.com/SKTBrain/KoBERT.git

cd KoBERT

echo "\n###### Step2 ######\n"
pip install .
pip install -r requirements.txt
pip install mxnet-cu101
pip install gluonnlp pandas tqdm
pip install transformers==2.1.1
pip install torch==1.3.1

cd ..

