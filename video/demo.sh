#!/bin/bash

split=( "s1" "s2" "s3" "s4" )

outdir="./data_utils/outputs/"
reltime="--reltime" # If its relative

cls_num=100
opt='adam'
epochs=40
loss='paper'
testset='test'

erls=( 1e-3 )
lrs=( 1e-4 )

# Loop over actions
for erl in ${erls[@]}
do
    for lr in ${lrs[@]}
    do
        for s in "${split[@]}"
        do
            runname="lr${lr}_erlambda${erl}_split${s}"
            # Random run
            python3 data_utils/bf_train.py ${reltime} --cls_num 0 --lr ${lr} --epochs 0 --opt ${opt} --loss ${loss} --model_name "mlprelu" --subaction all --split ${s} --testset_name ${testset}

            # Reg
            python3 data_utils/bf_train.py ${reltime} --cls_num 0 --lr ${lr} --epochs ${epochs} --opt ${opt} --loss ${loss} --model_name "mlprelu" --subaction all --split ${s} --testset_name ${testset}

            # Reg+Cls
            python3 data_utils/bf_train.py ${reltime} --erlambda ${erl} --cls_num ${cls_num} --lr ${lr} --epochs ${epochs} --opt ${opt} --loss ${loss} --model_name "mlprelu" --subaction all --split ${s} --testset_name ${testset}
        done # Over data splits
    done # Over lrs
done # Over lambda
