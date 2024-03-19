#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-7:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c10
#SBATCH -o /homes/mblum/outputs/stdout_train_%j_%t

source /homes/mblum/.bashrc
source /homes/mblum/miniconda3/etc/profile.d/conda.sh

conda activate literale

cd /homes/mblum/LiteralEvaluation
echo "start evaluation on YAGO3-10"

#python main_literal.py dataset YAGO3-10 epochs 0 process True --feature_type "numerical_literals.npy*train"
#python preprocess_num_lit.py --dataset YAGO3-10
#python preprocess_kg.py --dataset YAGO3-10


# --- Table: Numerical feature ablation: original features vs. random features ---
python -u main_literal.py dataset YAGO3-10 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluation/results/slurm_literaledistmult_yago3-10_org.txt
python -u main_literal.py dataset YAGO3-10 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/results/slurm_literaledistmult_yago3-10_rand.txt

python -u main_literal.py dataset YAGO3-10 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluation/results/slurm_literalecomplex_yago3-10_org.txt
python -u main_literal.py dataset YAGO3-10 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/results/slurm_literalecomplex_yago3-10_rand.txt

python -u main_kbln.py dataset YAGO3-10 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluation/results/slurm_kbln_yago3-10_org.txt
python -u main_kbln.py dataset YAGO3-10 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/results/slurm_kbln_yago3-10_rand.txt

python -u main_multitask.py dataset YAGO3-10 input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluation/results/slurm_multitask_yago3-10_org.txt
python -u main_multitask.py dataset YAGO3-10 input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/results/slurm_multitask_yago3-10_rand.txt
