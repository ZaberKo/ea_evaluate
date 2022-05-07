#!/bin/bash

set -ex

for iter in {10,50,100,150,200}
# for iter in {200,}
do
    declare -i iter
    # checkpoint_path="/root/workspace/model_data/ImpalaTrainer_2022-04-19_12-12-17/ImpalaTrainer_BreakoutNoFrameskip-v4_e6b7f_00000_0_2022-04-19_12-12-18/checkpoint_000$(printf %03d ${iter})/checkpoint-${iter}"
    # checkpoint_path="/root/workspace/model_data/ImpalaTrainer_2022-04-20_11-12-33/ImpalaTrainer_BreakoutNoFrameskip-v4_b887d_00000_0_2022-04-20_11-12-33/checkpoint_000$(printf %03d ${iter})/checkpoint-${iter}"
    checkpoint_path="/root/workspace/model_data/ImpalaTrainer_2022-04-20_15-50-27/ImpalaTrainer_BreakoutNoFrameskip-v4_8b518_00000_0_2022-04-20_15-50-27/checkpoint_000$(printf %03d ${iter})/checkpoint-${iter}"
    python evaluate_with_mutate.py ${checkpoint_path} --run ref.impala.ImpalaTrainer --episodes 1000 --config evaluate.yml --out eval_result_hard_20_15/mutate_result_${iter}.pkl
done