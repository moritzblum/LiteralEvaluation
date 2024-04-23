#!/bin/bash
mkdir saved_models
python wrangle_KG.py YAGO3-10
python wrangle_KG.py FB15k-237
python wrangle_KG.py FB15k
python wrangle_KG.py LitWD48K
python wrangle_KG.py Synthetic
python preprocess_kg.py --dataset FB15k-237
python preprocess_kg.py --dataset YAGO3-10
python preprocess_kg.py --dataset LitWD48K
python preprocess_kg.py --dataset Synthetic