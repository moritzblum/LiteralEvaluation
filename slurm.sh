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
echo "start evaluation"



#python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type numerical_literals_org.npy*train > ~/LiteralEvaluaton/slurm_1_kbln_org.txt
#python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type rand*train > ~/LiteralEvaluaton/slurm_2_kbln_rand.txt

#python -u main_multitask.py dataset FB15k-237 input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type numerical_literals_org.npy*train > ~/LiteralEvaluaton/slurm_3_multitask_org.txt
#python -u main_multitask.py dataset FB15k-237 input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type rand*train > ~/LiteralEvaluaton/slurm_4_multitask_rand.txt

#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.2" > ~/LiteralEvaluaton/slurm_5_ablation_02_org.txt
#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.2" > ~/LiteralEvaluaton/slurm_6_ablation_02_rand.txt

#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.4" > ~/LiteralEvaluaton/slurm_7_ablation_04_org.txt
#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.4" > ~/LiteralEvaluaton/slurm_8_ablation_04_rand.txt

#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.6" > ~/LiteralEvaluaton/slurm_9_ablation_06_org.txt
#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.6" > ~/LiteralEvaluaton/slurm_10_ablation_06_rand.txt


#python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_attr.npy*train" > ~/LiteralEvaluaton/slurm_17_kbln_literals_attr.txt
#python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_filtered_100.npy*train" > ~/LiteralEvaluaton/slurm_18_kbln_literals_filtered_100.txt
#python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "text_literals_clustered_100.npy*train" > ~/LiteralEvaluaton/slurm_19_kbln_literals_clustered_100.txt
#python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "zeros*train" > ~/LiteralEvaluaton/slurm_20_kbln_zeros.txt

#python -u main_literal.py dataset FB15k-237 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train" > ~/LiteralEvaluaton/slurm_21_complex_org.txt
#python -u main_literal.py dataset FB15k-237 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/slurm_22_complex_rand.txt

#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.1" > ~/LiteralEvaluaton/slurm_23_ablation_01_org.txt
#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.1" > ~/LiteralEvaluaton/slurm_24_ablation_01_rand.txt

#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.3" > ~/LiteralEvaluaton/slurm_25_ablation_03_org.txt
#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.3" > ~/LiteralEvaluaton/slurm_26_ablation_03_rand.txt

#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.5" > ~/LiteralEvaluaton/slurm_27_ablation_05_org.txt
#python -u main_literal.py dataset FB15k-237 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.5" > ~/LiteralEvaluaton/slurm_28_ablation_05_rand.txt










# --- Table 2 ---
# YAGO3-10
python main_literal.py dataset YAGO3-10 epochs 0 process True
python preprocess_num_lit.py --dataset YAGO3-10

python -u main_literal.py dataset YAGO3-10 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluaton/results/slurm_literaledistmult_yago3-10_org.txt
python -u main_literal.py dataset YAGO3-10 model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/LiteralEvaluaton/results/slurm_literaledistmult_yago3-10_rand.txt

python -u main_literal.py dataset YAGO3-10 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluaton/results/slurm_literalecomplex_yago3-10_org.txt
python -u main_literal.py dataset YAGO3-10 model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/LiteralEvaluaton/results/slurm_literalecomplex_yago3-10_rand.txt

python -u main_kbln.py dataset YAGO3-10 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train" > ~/LiteralEvaluaton/results/slurm_kbln_yago3-10_org.txt
python -u main_kbln.py dataset YAGO3-10 model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/results/slurm_kbln_yago3-10_rand.txt

python -u main_multitask.py dataset YAGO3-10 input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train" > ~/LiteralEvaluaton/results/slurm_multitask_yago3-10_org.txt
python -u main_multitask.py dataset YAGO3-10 input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/results/slurm_multitask_yago3-10_rand.txt


# LitWD48K
#python main_literal.py dataset LitWD48K epochs 0 process True --feature_type "numerical_literals.npy*train"
#python preprocess_num_lit.py --dataset LitWD48K

python -u main_literal.py dataset LitWD48K model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_decimal.npy*train" > ~/LiteralEvaluaton/results/slurm_literaledistmult_litwd48k_org.txt
python -u main_literal.py dataset LitWD48K model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/results/slurm_literaledistmult_litwd48k_rand.txt

python -u main_literal.py dataset LitWD48K model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_decimal.npy*train" > ~/LiteralEvaluaton/results/slurm_literalecomplex_litwd48k_org.txt
python -u main_literal.py dataset LitWD48K model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/results/slurm_literalecomplex_litwd48k_rand.txt

python -u main_kbln.py dataset LitWD48K model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_decimal.npy*train" > ~/LiteralEvaluaton/results/slurm_kbln_litwd48k_org.txt
python -u main_kbln.py dataset LitWD48K model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/results/slurm_kbln_litwd48k_rand.txt

python -u main_multitask.py dataset LitWD48K input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_decimal.npy*train" > ~/LiteralEvaluaton/results/slurm_multitask_litwd48k_org.txt
python -u main_multitask.py dataset LitWD48K input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train" > ~/LiteralEvaluaton/results/slurm_multitask_litwd48k_rand.txt



# --- Table 4 ---
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.1" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_01_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.1" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_01_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.2" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_02_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.2" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_02_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.3" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_03_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.3" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_03_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.4" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_04_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.4" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_04_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.5" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_05_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.5" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_05_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.6" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_06_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.6" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_06_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.7" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_07_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.7" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_07_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.8" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_08_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.8" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_08_rand.txt

python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals_org.npy*train_pruned_0.9" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_09_org.txt
python -u main_kbln.py dataset FB15k-237 model KBLN input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "rand*train_pruned_0.9" > ~/LiteralEvaluaton/results/slurm_ablation_kbln_09_rand.txt




# --- Table 5 ---
#python main_literal.py dataset Synthetic epochs 0 process True --feature_type "numerical_literals.npy*train"
#python preprocess_num_lit.py --dataset Synthetic

python -u main_literal.py dataset Synthetic model DistMult input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluaton/slurm_literaledistmult_synthetic_org.txt
python -u main_literal.py dataset Synthetic model ComplEx input_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluaton/slurm_literalecomplex_synthetic_org.txt
python -u main_kbln.py dataset Synthetic model KBLN input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluaton/slurm_kbln_synthetic_org.txt
python -u main_multitask.py dataset Synthetic input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True --feature_type "numerical_literals.npy*train" > ~/LiteralEvaluaton/slurm_multitask_synthetic_org.txt




















