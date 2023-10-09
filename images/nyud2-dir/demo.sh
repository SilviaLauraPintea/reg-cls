#!/bin/bash

DEVICES='0'         # Devices to run the exps on, preferably 2-4 GPUs
DATADIR='./data'    # Path to the data directory
BATCH=16            # Batch size should be 32
BACU=2              # Accumulate over multiple batches if needed
EPOCHS=5            # Number of epochs
LOSS="msecls"       # The losses to use during training
OPT='adam'          # The optimizer
LR=1e-4             # The learning rate
SEED=0              # The seed for this run
ERL=1.0             # The scaling between the losses
CLS=10              # The number of classes

CUDA_VISIBLE_DEVICES=${DEVICES} python3 train_cls.py \
        --data_dir ${DATADIR} \
        --seed ${SEED} \
        --losstype ${LOSS} \
        --reweight none \
        --batch_size ${BATCH} \
        --batch_acu ${BACU} \
        --epochs ${EPOCHS} \
        --cls_num ${CLS} \
        --opt ${OPT} \
        --erlambda ${ERL} \
        --lr ${LR} \
        --test_set "test" \
        --cls_equalize
