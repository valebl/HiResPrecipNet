#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict23_esp_0 #ict24_esp
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

## Testing
accelerate launch --config_file ${ACCELERATE_CONFIG_PATH} predictions.py ${USE_ACCELERATE} --input_path=${INPUT_PATH} --output_path=${OUTPUT_PATH} --log_file=${LOG_FILE} --graph_file=${GRAPH_FILE} --checkpoint_cl=${CHECKPOINT_CL} --checkpoint_reg=${CHECKPOINT_REG} --model_cl=${MODEL_CL} --model_reg=${MODEL_REG} --dataset_name=${DATASET_NAME} --output_file=${OUTPUT_FILE} --test_year_start=${TEST_YEAR_START} --test_month_start=${TEST_MONTH_START} --test_day_start=${TEST_DAY_START} --test_year_end=${TEST_YEAR_END} --test_month_end=${TEST_MONTH_END} --test_day_end=${TEST_DAY_END} --first_year=${FIRST_YEAR} --first_year_input=${FIRST_YEAR_INPUT} --batch_size=1
EOT
