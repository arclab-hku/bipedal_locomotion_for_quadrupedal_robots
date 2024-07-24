#!/bin/bash
CACHE=$1
python -B play_joy_45_imi.py --task=random_dog --s_flag=1 \
--algo=Base \
--priv_info \
--output_name=random_go1/Base/"${CACHE}" \
--checkpoint_model=last.pt \
--resume \
--resume_name=random_go1/Base/"${CACHE}" \
--export_policy




