#!/bin/bash
source $1
mkdir -p ${LOG_PATH}

sbatch << EOT
#!/bin/bash
#SBATCH -A ict23_esp_0
#SBATCH -p boost_usr_prod
#SBATCH --time ${TIME}       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=32   # 8 tasks out of 128
#SBATCH --job-name=${JOB_NAME}
# SBATCH --mail-type=FAIL,END
# SBATCH --mail-user=${MAIL}
#SBATCH -o ${LOG_PATH}/run.out
#SBATCH -e ${LOG_PATH}/run.err

#----------#
# PHASE 1  #
#----------#
module purge
module load --auto profile/meteo
module load cdo

source ${SOURCE_PATH}

cd ${INPUT_PATH_PHASE_1}

if [ ${PERFORM_PHASE_1} = true ] ; then
	source ${PHASE_1_PATH} ${LON_MIN} ${LON_MAX} ${LAT_MIN} ${LAT_MAX} ${INTERVAL} ${INPUT_PATH_PHASE_1} ${OUTPUT_PATH_PHASE_1} ${PREFIX_PHASE_1}
fi

#---------#
# PHASE 2 #
#---------#

if [ ${PERFORM_PHASE_2} = true ] ; then
	python3 ${PHASE_2_PATH} --input_path_phase_2=${INPUT_PATH_PHASE_2} --input_path_gripho=${INPUT_PATH_GRIPHO} --input_path_topo=${INPUT_PATH_TOPO} --gripho_file=${GRIPHO_FILE} --topo_file=${TOPO_FILE} --output_path=${OUTPUT_PATH_PHASE_2} --output_path_low=${OUTPUT_PATH_PHASE_2} --log_file=${LOG_FILE} --lon_min=${LON_MIN} --lon_max=${LON_MAX} --lat_min=${LAT_MIN} --lat_max=${LAT_MAX} --suffix=${SUFFIX_PHASE_2} ${LOAD_STATS} --stats_path=${STATS_PATH} --stats_file_high=${STATS_FILE_HIGH} --means_file_low=${MEANS_FILE_LOW} --stds_file_low=${STDS_FILE_LOW} --predictors_type=${PREDICTORS_TYPE} --lon_grid_radius_high=${LON_GRID_RADIUS_HIGH} --lat_grid_radius_high=${LAT_GRID_RADIUS_HIGH} --mean_std_over_variable_and_level_low
fi
EOT


