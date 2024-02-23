#!/bin/bash

#------------------#
# SLURM PARAMETERS #
#------------------#
TIME="00:01:00"
MEM="100G"
JOB_NAME="reg_GAN"
MAIL="vblasone@ictp.it"
LOG_PATH="/leonardo_work/ICT23_ESP_0/vblasone/HiResPrecipNet/runs/reg_GAN/"
N_GPU="1"

#--------------------------------------------#
# PATHS (conda env, main, accelerate config) #
#--------------------------------------------#
SOURCE_PATH="/leonardo/home/userexternal/vblasone/.bashrc"
ENV_PATH="/leonardo/home/userexternal/vblasone/.conda/envs/GNNenvT"
MAIN_PATH="/leonardo_work/ICT23_ESP_0/vblasone/HiResPrecipNet/"
ACCELERATE_CONFIG_PATH="/leonardo/home/userexternal/vblasone/.cache/huggingface/accelerate/default_config_1.yaml"

#-------------------------------#
# INPUT/OUTPUT FILES PARAMETERS #
#-------------------------------#
INPUT_PATH="/leonardo_work/ICT23_ESP_0/vblasone/HiResPrecipNet/Data/North/"
OUTPUT_PATH="/leonardo_work/ICT23_ESP_0/vblasone/HiResPrecipNet/runs/reg_GAN/"
LOG_FILE="log.txt"
TARGET_FILE="target_train_reg.pkl"
WEIGHTS_FILE="reg_weights.pkl"
GRAPH_FILE="low_high_graph.pkl"
OUT_CHECKPOINT_FILE="checkpoint.pth"
OUT_LOSS_FILE="loss.csv"
MODEL_TYPE="reg"
MODEL_NAME="HiResPrecipNet"
DATASET_NAME="Dataset_Graph"

#---------------------#
# TRAINING PARAMETERS #
#---------------------#

PCT_TRAINING=0.9
EPOCHS=210
BATCH_SIZE=16
LR_STEP_SIZE=100
LR=0.0001
WEIGHT_DECAY=0
LOSS_FN="weighted_mse_loss"
WANDB_PROJECT_NAME="reg_GAN"
#CHECKPOINT_CTD="/leonardo_work/ICT23_ESP_0/vblasone/HiResPrecipNet/runs/reg_north_fl_2/checkpoint_158.pth"

#----------------------------------------#
#             TRAINING PERIOD            #
#----------------------------------------#
TRAIN_YEAR_START=2001
TRAIN_MONTH_START=1
TRAIN_DAY_START=1
TRAIN_YEAR_END=2015
TRAIN_MONTH_END=11
TRAIN_DAY_END=30
FIRST_YEAR=2001

#----------------------------------------#
# COORDINATES OF CONSIDERED SPATIAL AREA #
#----------------------------------------#
LON_MIN=6.75
LON_MAX=14.00
LAT_MIN=43.75
LAT_MAX=47.00

#-----------------#
# BOOLEAN OPTIONS #
#-----------------#
CTD_TRAINING=false
FINE_TUNING=true
LARGE_GRAPH=true
USE_ACCELERATE=true

#--------------do not change here below------------------

if [ ${USE_ACCELERATE} = true ] ; then
	USE_ACCELERATE="--use_accelerate"
else
	USE_ACCELERATE="--no-use_accelerate"
fi

if [ ${CTD_TRAINING} = true ] ; then
	CTD_TRAINING="--ctd_training"
else
	CTD_TRAINING="--no-ctd_training"
fi

if [ ${FINE_TUNING} = true ] ; then
	FINE_TUNING="--fine_tuning"
else
	FINE_TUNING="--no-fine_tuning"
fi

if [ ${LARGE_GRAPH} = true ] ; then
	LARGE_GRAPH="--large_graph"
else
	LARGE_GRAPH="--no-large_graph"
fi

