#!/bin/bash

printf "========== Testing =========="
python GaussianBayes.py

python LogisticRegression.py

python MLP.py

python RandomForest.py

python KNeighbors.py
printf "\n========== Testing done =========\n"
