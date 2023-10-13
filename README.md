
# ReGMI

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


## MMNIST Experiments

Make sure there are 30GB free disk space for MMNIST storage.

Before experiments, use MMNIST/make_ds.py to produce MMNIST datasets first.

You can put MNIST raw dataset under directory "datasets" first.

```python make_ds.py```

It may cost several hours to produce MMNIST.

Run following command to inspect the accuracy float:

```python -m visdom.server```

Run following command to test different aggregators on MMNIST of different modes：

```python main.py --pooling rgp --bagsize 1```

optional pooling methods:
- rgp   (regressor-guided pooling)
- max   (max pooling)
- dsp   (dual stream pooling)
- abp   (attention-based pooling)
- gab   (gated attention-based pooling)
- tsp   (pooling from transmil using TPT module)


optional bag size:
1~64, multiple of 64, max 512

## UNBC Experiments

Make sure there are 800GB free disk space for UNBC storage.

Before test on UNBC, you need to redivide the UNBC dataset with 25 folds, generated under datasets/UNBC.

Run this command to test RGMIL on UNBC:

```python train_UNBC.py --pooling rgp --dataset UNBC```

It would cost a long time since dataset would be redivided and recreated. 

For all 25-fold cv, you may need more than a week on RTX 4090.





