# Numerical Literals in Link Prediction: A Critical Examination of Models and Datasets

# TODOs
* KGA mit Attriutive Ablation Methode Training überwachen und zum Paper hinzufügen 
* ACL Paper Format nach ISWC Paper Format überführen
* Paper auf Arxiv hochladen  
* Twitter Post zum angenommenen Paper

This repository contains the following resources for the paper [Numerical Literals in Link Prediction: 
A Critical Examination of Models and Datasets](todo Archiv Link):
* Source code to create the proposed datasets: Ablated datasets and Semi-Synthetic datasets. (`/preprocessing_literalevaluation`)
* Commands and settings to evaluate the investigated models on the datasets. (See below & `/slurm`)
* Training logs we obtained during the experiments. (`/data/results` & `/data/saved_models`)
* Jupyter Notebooks we used to create the visualizations and tables in the paper based on the training logs. (`/evaluation_notebooks`)

![alt text](https://anonymous.4open.science/api/repo/LiteralEvaluation-7545/file/data/tex/example_eiffel_tower.png)


## Create the Datasets
Please follow the instructions below to set up the `literalevaluation` conda environment to create the datasets.

```
conda create --name literalevaluation python=3.10
pip install -r requirements-literalevaluation.txt
```

**Ablated Datasets**

We propose ablated datasets to test the performance of a model if certain literal information or relational information
is removed from the dataset to evaluate the importance of the literal information. The ablated datasets are created from
existing literal containing Link Prediction datasets. All ablation strategies are described in the paper.

You can create the ablated datasets by running the following command:
```
python preprocessing_literalevaluation/get_literalevaluation_dataset.py --dataset {FB15k-237, YAGO3-10, LitWD48K} --features {org, rand, attribute} --literal_replacement_mode {sparse, full} --relation_ablation {0..10} --max_num_attributive_relations {1..} --data_dir {data path} --run_id {run id}
```

For example, to create the ablated dataset of FB15k-237 where every entity is assigned all attributes at random 
and 20% of the relational triples are ablated, run the following command:
```
python preprocessing_literalevaluation/get_literalevaluation_dataset.py --dataset FB15k-237 --features "rand" --literal_replacement_mode "full" --relation_ablation 20 --data_dir "data"
```
The new dataset will be stored as `data/FB15k-237_rand_rel-20_full`.




**Semi-Synthetic dataset**

We propose semi-synthetic datasets to test the performance of a model if literal values of entities are needed to be 
understood to make correct predictions. The `Synthetic` dataset we use for the evaluation in our paper is generated 
from the FB15k-237 dataset. 

We provide the `Synthetic` dataset in the `data` directory.

If you like to re-create the `Synthetic` dataset, or if you like to create
a semi-synthetic version of another dataset, you can use the 
`preprocessing_literalevaluation/get_synthetic_dataset.py` script.

```
python preprocessing_literalevaluation/get_synthetic_dataset.py --dataset_name {new semi-synthetic dataset name} --dataset {FB15k-237, YAGO3-10, LitWD48K} --class_mapping_file {class mapping file} --relevant_class_label {relevant class label}
```

For example, to create the semi-synthetic dataset `Synthetic` based on the FB15k-237 dataset (the one used in our experiments), run the following command:
```
python preprocessing_literalevaluation/get_synthetic_dataset.py --dataset_name Synthetic --dataset "FB15k-237" --class_mapping_file "data/FB15k-237/FB15k-237_class_mapping.csv" --relevant_class_label "human"
```


## Re-creating the Evaluation

### LiteralE
Please follow the instructions below to set up the `literale` conda environment.
```
conda create --name literale python=3.6.13
pip install -r requirements-literale.txt
conda activate literale
git clone git@github.com:SmartDataAnalytics/LiteralE.git
cd LiteralE
```

Place the datasets in the `data` directory and preprocess the datasets to evaluate. 
We use the placeholder `[eval-datadet]` for the dataset.
1. `mkdir saved_models`
2. `python wrangle_KG.py [eval-datadet]`
3. `python preprocess_kg.py --dataset [eval-datadet]`
4. Preprocess attributive data by consecutively running the following commands:
    1. Create vocab file: `python main_literal.py dataset [eval-datadet] epochs 0 process True`
    2. Numerical literals: `python preprocess_num_lit.py --dataset [eval-datadet]`
   
Then you can train the model by running the following command:
```
python main_literal.py dataset [eval-datadet] model {DistMult, ComplEx} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True
```

_Note:_  
The code only support computation on GPU (CUDA). The code can be adjusted easily to run on CPU by removing the `.cuda()` calls.


### TransEA
Please follow the instructions below to set up the `transea` conda environment.

```
conda create --name transea python=3.10
pip install -r requirements-transea.txt
conda activate transea
```

Then you can train the model by running the following command:
```
python transea/main_transea.py --model transea --dataset [eval-datadet] --alpha 0.3 --lr 0.001 --hidden 100 --saved_models_path="./saved_models"
```


### KGA
Please follow the instructions below to set up the `kga` conda environment.
```
conda create --name kga python=3.8
conda activate kga
pip install torch
pip install numpy
pip install pandas 
git clone git@github.com:Otamio/KGA.git
cd KGA
```

Then you can apply the literal transformations and train the model by running the following command: 
```
python augment_lp.py --mode "Quantile_Hierarchy" --bins 32 --dataset [eval-datadet]
python run.py --dataset [eval-datadet]_QHC_5 --model {distmult, tucker}
```


### Reproduce all Experiments
The scripts in the `slurm` directory contain the commands to reproduce all experiments. Simply run the following
scripts on a slurm cluster:
* `./slurm/slurm_FB15k-237.sh` (LiteralE)
* `./slurm/slurm_LitWD48K.sh` (LiteralE)
* `./slurm/slurm_YAGO3-10.sh` (LiteralE)
* `./slurm/slurm_Synthetic.sh` (LiteralE)
* `./slurm/slurm_TransEA.sh` (TransEA)
* `./slurm/slurm_KGA.sh` (KGA)


## Visualizing the Results
To visualize our results, you can use the Jupyter Notebooks in the `evaluation_notebooks` directory.
* `evaluation_notebooks/evaluate_ablations.ipynb` - Plot results and creates Latex tables for all ablation experiments.
* `evaluation_notebooks/evaluate_synthetic.ipynb` - Implements the Acc metric, plots the results, creates Latex tables for the experiments around the semi-synthetic dataset.


## Credits
For the evaluation of the models, we used the following repositories:
* LiteralE: [https://github.com/SmartDataAnalytics/LiteralE](https://github.com/SmartDataAnalytics/LiteralE)
* KGA: [https://github.com/Otamio/KGA](https://github.com/Otamio/KGA)
* Our TransEA is based on: [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.TransE.html)


## Cite

Please cite [our paper](todo archiv link) if you use our evaluation dataset in your own work:

```
@inproceedings{mblum-etal-2024-literalevaluation,
    title = "Numerical Literals in Link Prediction: A Critical Examination of Models and Datasets",
    author = "Blum, Moritz  and
      Ell, Basil  and
      Ill, Hannes and 
      Cimiano, Philipp",
    booktitle = "Proceedings of the 23rd International Semantic Web Conference (ISWC 2024)",
    month = November,
    year = "2024"
}
```