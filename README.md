# CSRNet
A pytorch [CSRNET](https://arxiv.org/abs/1802.10062) implementation for image crowd counting.  

Current naming support [shanghaitech dataset](https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map).

In order to run:

1. put the shanghai dataset under /datasets folder in the root path of the repo

2. Set absolute paths in [presets.py](presets.py)

3. preprocess density maps by running [prepare_dataset.py](prepare_dataset.py)

4. run [train.py](train.py) script

5. evaluate your model with [eval.py](eval.py)

With the example preset you can train a sub-70 MAE on ShanghaiTech_A_test.
