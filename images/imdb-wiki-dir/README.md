# Third party code from:

### **Delving into Deep Imbalanced Regression**
*Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi
38th International Conference on Machine Learning (ICML 2021)*

- Github code: https://github.com/YyzHarry/imbalanced-regression

**All rights remain with the authors.**

The only files added/modified are:

- ```train_cls.py``` -- Added option to also train/evaluate the model with a classification loss.
- ```dataset.py``` -- Added qunatification of the depth targets into a predefined number of classes.
- ```histeq.py``` -- Defines class ranges such that the class histograms are equalized.
- ```utils.py``` -- Added accuracy evaluation and histogram plotting routines.
- ```resnet.py``` --  Added class <Rcls> with a regression head and a classification head.

------------------------------------

# IMDB-WIKI-DIR
## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. __(Optional)__ We have provided required IMDB-WIKI-DIR meta file `imdb_wiki.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `train_cls.py`: main training and evaluation script
- `create_imdb_wiki.py`: create IMDB-WIKI raw meta data
- `preprocess_imdb_wiki.py`: create IMDB-WIKI-DIR meta file `imdb_wiki.csv` with balanced val/test set

#### Train a vanilla model

```bash
CUDA_VISIBLE_DEVICES=<DEVICES_LIST> python3 train_cls.py \
    --losstype <'mse' OR 'msecls'> \
    --data_dir <PATH_TO_DATADIR> \
    --reweight none \
    --batch_size <BATCH_SIZE> \
    --epoch <NR_EPOCH> \
    --cls_equalize \ # For equalized classes
    --balance_data \ # For balanced training
    --cls_num <NR_CLASSES>
```

Always specify `CUDA_VISIBLE_DEVICES` for GPU IDs to be used (by default, 4 GPUs) and `--data_dir` when training a model or directly fix your default data directory path in the code. We will omit these arguments in the following for simplicity.

#### Evaluate a trained checkpoint

```bash
python3 train_cls.py [...evaluation model arguments...] \
    --evaluate
    --resume <path_to_evaluation_ckpt>
    --test_set <'test' OR 'val'>
    --losstype <'mse' OR 'msecls'> \
    --cls_equalize \ # For equalized classes
    --cls_num <NR_CLASSES>
```
