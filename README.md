
# RGMIL

python version : 3.10

Requirements:
- numpy
- scikit-learn
- torch
- torchvision
- scipy
- PIL
- visdom
- pandas
- pingouin

## Benchmark Experiments

All the benchmark datasets are uploaded to datasets/Benchmark directory.
Run following command to test RGP on these datasets：

```python benchmark.py```

You can also modify the **DATASETS** array in benchmark.py to perform specific experiments on different benchmark datasets.

Run following command to test DSMIL on these datasets for comparison：

```python train_mil.py```

You can also modify the **DATASETS** array in train_mil.py to perform specific experiments on different benchmark datasets.



## UNBC Experiments
Make sure there are 800GB free disk space for UNBC storage.

Before test on UNBC, put the UNBC original dataset under datasets/UNBC.

Run this command to test RGMIL on UNBC:

```python train_UNBC.py --pooling rgp --dataset UNBC```

It would cost a long time since dataset would be divided and recreated.




