#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-7:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c10
#SBATCH -o /homes/mblum/outputs/stdout_LiteralEvaluation_attribute_KGA_%A_%a.txt
#SBATCH --array=0-24%10

source /homes/mblum/.bashrc
source /homes/mblum/miniconda3/etc/profile.d/conda.sh
conda activate tucker
cd /homes/mblum/KGA

job_count=0

literal_replacement_mode="sparse"

for run_id in "1" "2" "3"; do
  for dataset in "LitWD48K" "YAGO3-10" "FB15k-237" "Synthetic"; do

    for features in "attribute"; do  # todo "rand" "org" "attribute"
      kga_mode="Quantile_Hierarchy"

      # --- new ---

      # test
      #python augment/augment_lp.py --dataset "FB15k-237" --mode "Quantile_Hierarchy" --bins 32 --literalevaluation --relation_ablation "0" --features "attribute" --run_id "0+distmult"
      #python run.py --dataset "FB15k-237_attribute_rel-0_full_run-0+distmult_QHC_5" --model "distmult" --output out_test
      #python augment/clean_data_directory.py --dataset "FB15k-237" --relation_ablation "0" --features "attribute" --run_id "0+distmult"


      if [ "${features}" == "attribute" ]; then

        for model in "tucker" "distmult"; do

          if [ "${job_count}" -ne "${SLURM_ARRAY_TASK_ID}" ]; then
            job_count=$((job_count+1))
            continue
          fi

          python augment/augment_lp.py --dataset "${dataset}" --mode "${kga_mode}" --bins 32 --literalevaluation --relation_ablation "0" --features "${features}" --run_id "${run_id}+${model}"
          kga_dataset_name="${dataset}_${features}_rel-0_${literal_replacement_mode}_run-${run_id}+${model}_QHC_5"
          echo "Train ${model} on ${kga_dataset_name}"
          python run.py --dataset "${kga_dataset_name}" --model "${model}" --output out_${run_id}
          python augment/clean_data_directory.py --dataset "${dataset}" --relation_ablation "0" --features "${features}" --run_id "${run_id}+${model}"

          exit 0
        done
      else
        exit 0 # todo move all code below here
      fi
      # --- end new ---


#      if [ "${dataset}" != "FB15k-237" ]; then
#        pruning=("0")
#      else
#        pruning=("0" "10" "20" "30" "40" "50" "60" "70" "80" "90")
#      fi

#      if [ "${dataset}" == "YAGO3-10" ]; then
#        rand_features_limit=3
#      else
#        rand_features_limit=20
#      fi

#      for pruned in "${pruning[@]}"; do
#        for model in "tucker" "distmult"; do

          #if [ "$dataset" != "Synthetic" ]; then
          #  python run.py --dataset "${dataset}" --model "${model}" --output "out_${run_id}"
          #fi

#          if [ "${job_count}" -ne "${SLURM_ARRAY_TASK_ID}" ]; then
#            job_count=$((job_count+1))
#            continue
#          fi

#          echo "${job_count} Evaluate ${model} on ${dataset} with ${features} features and ${pruned} relations pruned as run ${run_id} with rand_features_limit ${rand_features_limit}"
#          job_count=$((job_count+1))

          # preprocess dataset
#          python augment/augment_lp.py --dataset "${dataset}" --mode "${kga_mode}" --bins 32 --literalevaluation --relation_ablation "${pruned}" --features "${features}" --run_id "${run_id}+${model}" --rand_features_limit "${rand_features_limit}" --literal_replacement_mode "${literal_replacement_mode}"

          # train model
#          kga_dataset_name="${dataset}_${features}_rel-${pruned}_${literal_replacement_mode}_run-${run_id}+${model}_QHC_5"
#          echo "Train ${model} on ${kga_dataset_name}"
#          python run.py --dataset "${kga_dataset_name}" --model "${model}" --output out_${run_id}

          # delete dataset
#          python augment/clean_data_directory.py --dataset "${dataset}" --relation_ablation "${pruned}" --features "${features}" --literal_replacement_mode "${literal_replacement_mode}" --run_id "${run_id}+${model}"
#          echo "Evaluation completed and all files deleted."
#          exit 0
#        done
#      done
    done
  done
done
