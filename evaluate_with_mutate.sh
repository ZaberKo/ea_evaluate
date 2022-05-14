#!/bin/bash

set -ex

# for iter in {10,50,100,150,200}
# for iter in {100,200,300,400,500,600,700}
for iter in {700,}
do
    declare -i iter
    checkpoint_path="/root/workspace/model_data/ImpalaTrainer_2022-05-08_01-25-34/ImpalaTrainer_SpaceInvadersNoFrameskip-v4-TimeLimit40000_b432f_00000_0_2022-05-08_01-25-35/checkpoint_000$(printf %03d ${iter})/checkpoint-${iter}"
    python evaluate_with_mutate.py ${checkpoint_path} --run ref.impala.ImpalaTrainer --env "SpaceInvadersNoFrameskip-v4" --episodes 300 --mutate_nums 50 --config evaluate.yml --out "SpaceInvaders_01_25_without_conv_frac_05/mutate_result_${iter}.pkl"
done