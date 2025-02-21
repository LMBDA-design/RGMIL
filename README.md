
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


**UPDATE 2025-02-21:**

In the bag-level benchmark evaluation, our training and evaluation code comes from [DSMIL](https://github.com/binli123/dsmil-wsi), and the benchmark datasets are downloaded from online resources. This is the version we presented publicly. However, we recently noticed that the version of the dataset we downloaded (as provided in the `datasets/Benchmark` directory) may have an extra feature dimension compared to the original version (e.g., FOX dataset with 230+1 dimensions). We are unsure whether this additional feature causes information leakage.


In comparative experiments, please pay additional attention to the feature dimensions (it is recommended to re-implement **ALL** methods using the unified version of the dataset with consistent dimensions if you use the datasets we collected here). In the re-testing of the original dimension with a unified dataset, the model's bag-level performance remains SOTA level, but the performance may decrease!




## UNBC Experiments
Make sure there are 800GB free disk space for UNBC storage.

Before test on UNBC, put the UNBC original dataset under datasets/UNBC.

Run this command to test RGMIL on UNBC:

```python train_UNBC.py --pooling rgp --dataset UNBC```

It would cost a long time since dataset would be divided and recreated.




