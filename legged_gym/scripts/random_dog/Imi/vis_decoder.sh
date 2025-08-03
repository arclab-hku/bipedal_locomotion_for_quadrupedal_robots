#!/bin/bash
CACHE=$1
python -B play_decoder.py --task=bipedal_dog --s_flag=1 \
--algo=ImiDecoder \
--priv_info \
--output_name=random_dog/Imi/"${CACHE}" \
--checkpoint_model=best.pt \
--resume \
--resume_name=random_dog/Imi/"${CACHE}" \
--export_policy




