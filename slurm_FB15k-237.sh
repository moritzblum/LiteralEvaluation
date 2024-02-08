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
echo "start evaluation on FB15k-237"

python main_literal.py dataset FB15k-237 epochs 0 process True --feature_type "numerical_literals.npy*train"
python preprocess_num_lit.py --dataset FB15k-237
python preprocess_kg.py --dataset FB15k-237


# --- Table: Numerical feature ablation: original features vs. random features ---
python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluation/results/slurm_literaledistmult_fb15k-237_org.txt
python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/results/slurm_literaledistmult_fb15k-237_rand.txt

python -u main_literal.py dataset FB15k-237 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluation/results/slurm_literalecomplex_fb15k-237_org.txt
python -u main_literal.py dataset FB15k-237 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/results/slurm_literalecomplex_fb15k-237_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" >  ~/LiteralEvaluation/results/slurm_kbln_fb15k-237_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/slurm_kbln_fb15k-237_rand.txt

python -u main_multitask.py dataset FB15k-237 input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluation/slurm_multitask_fb15k-237_org.txt
python -u main_multitask.py dataset FB15k-237 input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluation/slurm_multitask_fb15k-237_rand.txt


# --- Table: Comparison of different features derived from the origial numerical features ---
python -u main_literal.py dataset FB15k-237 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_attr.npy*train" > ~/LiteralEvaluation/results/slurm_literalecomplex_fb15k-237_attr.txt
python -u main_literal.py dataset FB15k-237 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_filtered_100.npy*train" > ~/LiteralEvaluation/results/slurm_literalecomplex_fb15k-237_filtered-100.txt
python -u main_literal.py dataset FB15k-237 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "zeros*train" > ~/LiteralEvaluation/results/slurm_literalecomplex_fb15k-237_zeros.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_attr.npy*train" > ~/LiteralEvaluation/results/slurm_kbln_fb15k-237_attr.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_filtered_100.npy*train" > ~/LiteralEvaluation/results/slurm_kbln_fb15k-237_filtered-100.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "zeros*train" > ~/LiteralEvaluation/results/slurm_kbln_fb15k-237_zeros.txt


# --- Table: MRR score after removing a certain percentage of relational triples ---
declare -a arr=("10" "20" "30", "40", "50", "60", "70", "80", "90")
for i in "${arr[@]}"
do
  python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train_pruned_$i" > ~/LiteralEvaluation/slurm_literaledistmult_fb15k-237_org_$i.txt
  python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_$i" > ~/LiteralEvaluation/slurm_literaledistmult_fb15k-237_rand_$i.txt
done

for i in "${arr[@]}"
do
  python -u main_multitask.py dataset FB15k-237 input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train_pruned_$i" > ~/LiteralEvaluation/slurm_multitask_fb15k-237_org_$i.txt
  python -u main_multitask.py dataset FB15k-237 input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_$i" > ~/LiteralEvaluation/slurm_multitask_fb15k-237_rand_$i.txt
done







