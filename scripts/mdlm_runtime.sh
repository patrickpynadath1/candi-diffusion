#!/bin/bash


export HYDRA_FULL_ERROR=1


number_sample_batches=10
file_name=mldm_runtime_1024.txt
touch $file_name

for steps in 8 16 32 64 128 256 512 1024
do 
    for temp in .9
    do  
        echo "Steps: $steps, Temp: $temp" >> $file_name
        bash scripts/gen_ppl_owt_mdlm.sh $1 $steps $temp $number_sample_batches >> $file_name
    done
done