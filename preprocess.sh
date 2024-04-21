#!/bin/bash
mkdir data
mkdir data/YAGO3-10
mkdir data/FB15k-237
mkdir saved_models
tar -xvf datasets/YAGO3-10.tar.gz -C data/YAGO3-10
tar -xvf datasets/FB15k-237.tar.gz -C data/FB15k-237
python wrangle_KG.py YAGO3-10
python wrangle_KG.py FB15k-237
python wrangle_KG.py FB15k
python wrangle_KG.py LitWD48K
python wrangle_KG.py Synthetic
python preprocess_kg.py --dataset FB15k-237
python preprocess_kg.py --dataset YAGO3-10
python preprocess_kg.py --dataset LitWD48K
python preprocess_kg.py --dataset Synthetic