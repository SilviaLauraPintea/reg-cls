#!/bin/bash

####################################################################################
#### Step 2: train the models ######################################################
####################################################################################
#### /// Edit the parameters here as needed /// ####################################
####################################################################################

datapath="data/"

# Pick here the dataset to run experiments on
datadir=( "clean" "noisey_0.1" "ood")

# Pick the seeds to run experiments over
seeds=( 0 421 8125 2481 849 )

# The value of lambda per dataset
erlambda=( 1e+2 1e+3 1e+4 )

# The learning rate
lrs=( 1e-3 1e-2 1e-4 )

# The dataset to run experiments on: val/test
evalset='test'

#-------------- Start the actual runs ---------------------------------------------#
idx=0
# Loop over dataset settings
for d in "${datadir[@]}"
do
    erl=${erlambda[idx]}
    lr=${lrs[idx]}
    logfile="${evalset}_lambda_${erl}.pkl"
    echo "Log file ${logfile}"
    echo "Data dir: ${d}"

    k=0
    # Loop over functions
    while [ $k -ne 10 ]
    do

        i=0
        # Loop over random seeds
        for s in "${seeds[@]}"
        do
            echo "Seed: ${s}"

            # Loop over classes
            for j in {2..10..2}
            do
                cls=$((1<<$j))
                echo "Class: ${cls}"
                python main.py --max_epochs 80 --erlambda ${erl} --lr ${lr} --num-segment ${cls} --random-seed ${s} --log-file "logs/${d}/${logfile}" --name "severe_${i}_${d}_fct${k}_lambda" --path-train "${datapath}/${d}/severe${i}fct${k}_train.pkl" --path-val "${datapath}/${d}/uniform${i}fct${k}_val.pkl" --path-test "${datapath}/${d}/uniform${i}fct${k}_${evalset}.pkl"
                python main.py --max_epochs 80 --erlambda ${erl} --lr ${lr} --num-segment ${cls} --random-seed ${s} --log-file "logs/${d}/${logfile}" --name "moderate_${i}_${d}_fct${k}_lambda" --path-train "${datapath}/${d}/moderate${i}fct${k}_train.pkl" --path-val "${datapath}/${d}/moderate${i}fct${k}_val.pkl" --path-test "${datapath}/${d}/uniform${i}fct${k}_${evalset}.pkl"
                python main.py --max_epochs 80 --erlambda ${erl} --lr ${lr} --num-segment ${cls} --random-seed ${s} --log-file "logs/${d}/${logfile}" --name "mild_${i}_${d}_fct${k}_lambda" --path-train "${datapath}/${d}/mild${i}fct${k}_train.pkl" --path-val "${datapath}/${d}/mild${i}fct${k}_val.pkl" --path-test "${datapath}/${d}/uniform${i}fct${k}_${evalset}.pkl"
                python main.py --max_epochs 80 --erlambda ${erl} --lr ${lr} --num-segment ${cls} --random-seed ${s} --log-file "logs/${d}/${logfile}" --name "uniform_${i}_${d}_fct${k}_lambda" --path-train "${datapath}/${d}/uniform${i}fct${k}_train.pkl" --path-val "${datapath}/${d}/uniform${i}fct${k}_val.pkl" --path-test "${datapath}/${d}/uniform${i}fct${k}_${evalset}.pkl"
            done # over classes
            ((i=i+1))

        done # over seeds
        ((k=k+1))

    done # over functions
    ((idx=idx+1))

done # over datasets
