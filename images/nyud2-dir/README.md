# Third party code from:

### **Delving into Deep Imbalanced Regression**
*Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi
38th International Conference on Machine Learning (ICML 2021)*

- Github code: https://github.com/YyzHarry/imbalanced-regression

**All rights remain with the authors.**

The only files added/modified are:

- ```train_cls.py``` -- Added option to also train the model with a classification loss.
- ```test_cls.py``` -- Added option to also evaluate the model trained with a classification head.
- ```loaddata.py``` -- Added qunatification of the depth targets into a predefined number of classes.
- ```histeq.py``` -- Defines class ranges such that the class histograms are equalized.
- ```util.py``` -- Added accuracy evaluation and histogram plotting routines.
- ```models/```
  - ```models/modules.py``` -- Added class <Rcls> with a regression head and a classification head.
  - ```models/net.py``` -- Integrated the <Rcls> class into the model definition.

------------------------------------


# NYUD2-DIR
## Installation

#### Prerequisites

1. Download and extract NYU v2 dataset to folder `./data` using

```bash
python download_nyud2.py
```

2. __(Optional)__ We have provided required meta files `nyu2_train_FDS_subset.csv` and `test_balanced_mask.npy`  for efficient FDS feature statistics computation and balanced test set mask in folder `./data`. To reproduce the results in the paper, please directly use these two files. If you want to try different FDS computation subsets and balanced test set masks, you can run

```bash
python preprocess_nyud2.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- numpy, pandas, scipy, tqdm, matplotlib, PIL, gdown, tensorboardX
- pickle5

## Code Overview

#### Main Files

- `train_cls.py`: main training script
- `test_cls.py`: main evaluation script
- `preprocess_nyud2.py`: create meta files `nyu2_train_FDS_subset.csv` and `test_balanced_mask.npy` for NYUD2-DIR

## Getting Started

#### Train a vanilla model

```bash
CUDA_VISIBLE_DEVICES=<DEVICE_LIST> python3 train_cls.py \
    --losstype <'mse' OR 'msecls'> \
    --data_dir <PATH_TO_DATADIR> \
    --reweight none \
    --batch_size <BATCH_SIZE> \
    --epoch <NR_EPOCHS> \
    --cls_equalize \ # For equalized classes
    --balance_data \ # For balanced training
    --cls_num <NR_CLASSES>
```

Always specify `CUDA_VISIBLE_DEVICES` for GPU IDs to be used (by default, 4 GPUs) and `--data_dir` when training a model or directly fix your default data directory path in the code. We will omit these arguments in the following for simplicity.

#### Evaluate a trained checkpoint

```bash
python3 test_cls.py \
    --data_dir <path_to_data_dir>
    --eval_model <path_to_evaluation_ckpt>
    --test_set <'test' OR 'val'>
    --losstype <'mse' OR 'msecls'> \
    --cls_equalize \ # For equalized classes
    --cls_num <NR_CLASSES>
```
