#5!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict24_esp
#SBATCH -p boost_usr_prod
#SBATCH --time=${TIME}       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=32 # out of 128
#SBATCH --gres=gpu:${N_GPU}       # 1 gpus per node out of 4
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

source ${SOURCE_PATH}

#source ~/anaconda/etc/profile.d/conda.sh
conda activate ${ENV_PATH}

cd ${MAIN_PATH}

## training
accelerate launch --config_file ${ACCELERATE_CONFIG_PATH} main_GAN.py --input_path=${INPUT_PATH} --output_path=${OUTPUT_PATH} --log_file=${LOG_FILE} --target_file=${TARGET_FILE} --weights_file=${WEIGHTS_FILE} --graph_file=${GRAPH_FILE} --out_checkpoint_file=${OUT_CHECKPOINT_FILE} --out_loss_file=${OUT_LOSS_FILE} --pct_trainset=${PCT_TRAINING} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} --step_size=${LR_STEP_SIZE} --lr=${LR} --weight_decay=${WEIGHT_DECAY} --loss_fn=${LOSS_FN} --model_type=${MODEL_TYPE} --model_name=${MODEL_NAME} --dataset_name=${DATASET_NAME} --wandb_project_name=${WANDB_PROJECT_NAME} --lon_min=${LON_MIN} --lon_max=${LON_MAX} --lat_min=${LAT_MIN} --lat_max=${LAT_MAX} ${USE_ACCELERATE} ${CTD_TRAINING} ${FINE_TUNING} --train_year_start=${TRAIN_YEAR_START} --train_month_start=${TRAIN_MONTH_START} --train_day_start=${TRAIN_DAY_START} --train_year_end=${TRAIN_YEAR_END} --train_month_end=${TRAIN_MONTH_END} --train_day_end=${TRAIN_DAY_END} --first_year=${FIRST_YEAR} --checkpoint_ctd=${CHECKPOINT_CTD} --out_checkpoint_file=${OUT_CHECKPOINT_FILE}
EOT

