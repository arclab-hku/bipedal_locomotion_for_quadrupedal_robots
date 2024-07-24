#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
RESUMENAME=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python -B train.py --task=bipedal_dog  --num_envs=1024 --headless --seed=${SEED} \
--s_flag=1 \
--algo=Imi \
--priv_info \
--output_name=random_dog/Imi/"${CACHE}" \
--checkpoint_model=last.pt \
--resume \
--resume_name=random_dog/Imi/"${RESUMENAME}" \
${EXTRA_ARGS}
