# LiteralE

Evaluation of Link Prediction Models that incorporate numerical literals.

This repository contains the source code, datasets, training result logs, and visualization Jupyter Notebooks. 


<p align="center">
<img src="./data/tex/example_eiffel_tower.png" width="400">
</p>

<p align="center">
Fig. 1: Example Knowledge Graph with numerical literals.
</p>

### Credits

This project is constructed upon the foundation laid by Agustinus Kristiadi's codebase 
available at <https://github.com/SmartDataAnalytics/LiteralE>, which in turn relies on the ConvE implementation 
by Tim Dettmers, accessible at <https://github.com/TimDettmers/ConvE>.


### ToDos
* @Hannes implement command line parameters: `python create_variations.py --dataset {FB15k-237, YAGO3-10, LitWD48K}`
* @Hannes add description to `dataset_statistics.ipynb` and reference table 1
* @Moritz describe how to create the synthetic dataset
* @Moritz create script `python create_ablation.py --dataset {FB15k-237, YAGO3-10, LitWD48K, Synthetic}` from create_relational_ablation_sets.ipynb
* @Moritz put FB15k-237_class_mapping.csv into an appropriate folder 
* @Moritz integrate TranEA code
* @Moritz check where datasets are provided 


### Getting Started

#### Set-up a Conda Environment and Install Dependencies

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

In the following, we will indicate which environment to use for each experiment by leading (literale) or (transea).

Note: We only support computation on GPU (CUDA). The code can be adjusted easily to run on CPU by removing the `.cuda()` calls.

#### Create Synthetic Dataset
TODO

#### Preprocess datasets
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
