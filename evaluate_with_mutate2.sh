#!/bin/bash

set -exo pipefail

# for iter in {10,50,100,150,200}
# for iter in {100,300,500,700,900}
# do
#     declare -i iter
#     checkpoint_path="/root/share_workspace/model/ImpalaTrainer_2022-05-09_21-32-28/ImpalaTrainer_QbertNoFrameskip-v4-TimeLimit18000_7889f_00000_0_2022-05-09_21-32-28/checkpoint_000$(printf %03d ${iter})/checkpoint-${iter}"
#     python evaluate_with_mutate.py ${checkpoint_path} --run ref.impala.ImpalaTrainer --env "QbertNoFrameskip-v4" --episodes 300 --mutate_nums 50 --config evaluate-cpu.yml --out "eval_result/Qbert_21_32_without_conv_frac_01_new/mutate_result_${iter}.pkl"
# done



# for iter in {100,300,500,700,900}
# for iter in {10..200..20}
for iter in {150,200,250,300}
do
    declare -i iter
    checkpoint_path="/root/share_workspace/model/ImpalaTrainer_2022-05-10_18-05-22/ImpalaTrainer_BeamRiderNoFrameskip-v4-TimeLimit18000_b4a6a_00000_0_2022-05-10_18-05-23/checkpoint_000$(printf %03d ${iter})/checkpoint-${iter}"
    python evaluate_with_mutate.py ${checkpoint_path} --run ref.impala.ImpalaTrainer --env "BeamRiderNoFrameskip-v4" --episodes 300 --mutate_nums 20 --config evaluate-cpu.yml --out "eval_result/BeamRider_18_05_without_conv_frac_01_new/mutate_result_${iter}.pkl"
done