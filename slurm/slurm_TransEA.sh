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

conda activate lp 

cd /homes/mblum/LiteralEvaluation
echo "start evaluation of TransEA"

python -u main_transea.py --model transea --dataset FB15k-237 --alpha 0.3 --lr 0.001 --hidden 100 --features numerical_literals.npy --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_fb15k-237_org.txt
python -u main_transea.py --model transea --dataset LitWD48K --alpha 0.3 --lr 0.001 --hidden 100 --features numerical_literals_decimal.npy --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_litwd48k_org.txt
python -u main_transea.py --model transea --dataset YAGO3-10 --alpha 0.3 --lr 0.001 --hidden 100 --features numerical_literals.npy --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_yago3-10_org.txt
python -u main_transea.py --model transea --dataset Synthetic --alpha 0.3 --lr 0.001 --hidden 100 --features numerical_literals.npy --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_synthetic_org.txt

python -u main_transea.py --model transea --dataset FB15k-237 --alpha 0.3 --lr 0.001 --hidden 100 --features rand --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_fb15k-237_rand.txt
python -u main_transea.py --model transea --dataset LitWD48K --alpha 0.3 --lr 0.001 --hidden 100 --features rand --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_litwd48k_rand.txt
python -u main_transea.py --model transea --dataset YAGO3-10 --alpha 0.3 --lr 0.001 --hidden 100 --features rand --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_yago3-10_rand.txt
python -u main_transea.py --model transea --dataset Synthetic --alpha 0.3 --lr 0.001 --hidden 100 --features rand --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_synthetic_rand.txt

