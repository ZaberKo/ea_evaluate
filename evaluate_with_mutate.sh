#!/bin/bash

set -exo pipefail

# for iter in {10,50,100,150,200}
for iter in {100,200,300,400}
do
    declare -i iter
    checkpoint_path="/root/share_workspace/model/ImpalaTrainer_2022-04-27_20-43-00/ImpalaTrainer_BreakoutNoFrameskip-v4_925da_00000_0_2022-04-27_20-43-00/checkpoint_000$(printf %03d ${iter})/checkpoint-${iter}"
    python evaluate_with_mutate_tmp.py ${checkpoint_path} --run ref.impala.ImpalaTrainer --env "BreakoutNoFrameskip-v4" --episodes 300 --mutate_nums 50 --config evaluate-cpu.yml --out "eval_result/Breakout_20_43_without_conv_frac_01/mutate_result_${iter}.pkl"
done


