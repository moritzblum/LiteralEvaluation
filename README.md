# Numerical Literals in Link Prediction: A Critical Examination of Models and Datasets

This repository contains the source code, datasets, training result logs, and visualization Jupyter Notebooks 
associated with the paper "Numerical Literals in Link Prediction: A Critical Examination of Models and Datasets". 

You can find all training result logs used for the paper in `data/results` and the 
Jupyter Notebooks that created all Visualizations and Tables in `evaluation_notebooks`.

![alt text](https://anonymous.4open.science/api/repo/LiteralEvaluation-7545/file/data/tex/example_eiffel_tower.png)


### Getting Started
Please follow the instructions below to set up the environment and datasets to reproduce the experiments.

#### 1. Set-up a Conda Environment and Install Dependencies
Set up the environment for all experiments (except TransEA) by running the following commands:
```
conda create --name literale python=3.6.13
pip install -r requirements-literale.txt
```

For TransEA, set up the environment by running the following commands:
```
conda create --name transea python=3.10
pip install -r requirements-transea.txt
```

_Note:_  
* In the following, we will indicate which environment to use for each experiment by leading (literale) or (transea).
* We only support computation on GPU (CUDA). The code can be adjusted easily to run on CPU by removing the `.cuda()` calls.

#### 2. Create datasets 
_Important:_ 
* These steps are optional, as the datasets are already provided in the `data` directory.
* Use the `literale` environment.

Create the Semi-Synthetic Dataset: The Synthetic dataset is generated from the FB15k-237 dataset. 
The dataset is generated to test the performance of the model if both, 
structure and literal values of entities, are needed to be understood in order 
to make correct predictions. Run `python create_synthetic_dataset.py` to generate the synthetic dataset.

Create the relational ablation Datasets: The ablation datasets are created from existing literal containing Link 
Prediction datasets by removing certain relational relations from the original dataset. 
Run `python create_relational_ablations.py {FB15k-237, YAGO3-10, LitWD48K}` to generate the ablation datasets. 
In the paper we only evaluate the ablation datasets for the FB15k-237 dataset.

#### 3. Preprocess datasets 
_Important:_ Use the `literale` environment.
1. Preprocess relational data by running: `chmod +x preprocess.sh && ./preprocess.sh`
2. Preprocess attributive data by consecutively running the following commands:
    1. Create vocab file: `python main_literal.py dataset {FB15k-237, YAGO3-10, LitWD48K, Synthetic} epochs 0 process True`
    2. Numerical literals: `python preprocess_num_lit.py --dataset {FB15k-237, YAGO3-10, LitWD48K, Synthetic}`
    3. Variations of numerical literals: `python create_variations.py --dataset {FB15k-237, YAGO3-10, LitWD48K}`
3. Create the relational triple ablation dataset by running: `python create_ablation.py --dataset {FB15k-237, YAGO3-10, LitWD48K, Synthetic}`
   

### Reproduce all Experiments
The shell scripts in the `slurm` directory contain the commands to reproduce all experiments. Simply run the following
commands to reproduce the experiments for the respective datasets/models:
* `./slurm/slurm_FB15k-237.sh` (use the `literale` environment)
* `./slurm/slurm_LitWD48K.sh` (use the `literale` environment)
* `./slurm/slurm_YAGO3-10.sh` (use the `literale` environment)
* `./slurm/slurm_Synthetic.sh` (use the `literale` environment)
* `./slurm/slurm_TransEA.sh` (use the `transea` environment)

The results will be saved in the `results` directory. The slurm scripts can be run on a GPU cluster by starting the 
scripts with `sbatch`. Attention: make sure the specified cluster node resources are available.


### Reproducing Individual Runs
You can run all commands in the shell scripts manually. Make sure to run the pre-processing steps called by the scripts first.


### Visualizing the Results
To visualize the results, you can use the Jupyter Notebooks in the `evaluation_notebooks` directory.
* `evaluation_notebooks/evaluate_ablations.ipynb` - Plot results and creates Latex tables for all ablation experiments.
* `evaluation_notebooks/evaluate_synthetic.ipynb` - Implements the Acc metric, plots the results, creates Latex tables for the experiments around the semi-synthetic dataset.

In the Notebooks it is specified which result files are needed. We provide the result files of our experiments in the `results` and `saved_models` directory.


### Credits

This project is constructed upon the foundation laid by Agustinus Kristiadi's codebase 
available at <https://github.com/SmartDataAnalytics/LiteralE>, which in turn relies on the ConvE implementation 
by Tim Dettmers, accessible at <https://github.com/TimDettmers/ConvE>.