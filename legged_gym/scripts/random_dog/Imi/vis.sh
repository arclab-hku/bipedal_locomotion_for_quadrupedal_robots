#!/bin/bash
CACHE=$1
python -B play_imi.py --task=bipedal_dog --s_flag=1 \
--algo=Imi \
--priv_info \
--output_name=random_dog/Imi/"${CACHE}" \
--checkpoint_model=last.pt \
--resume \
--resume_name=random_dog/Imi/"${CACHE}" \
--export_policy




