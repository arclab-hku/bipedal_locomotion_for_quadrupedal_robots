#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python -B train.py --task=bipedal_dog  --num_envs=1024 --headless --seed=${SEED} \
--algo=Imi \
--priv_info \
--output_name=random_dog/Imi/"${CACHE}" \
${EXTRA_ARGS}

