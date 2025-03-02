
# [RGMIL: Guide Your Multiple-Instance Learning Model with Regressor](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html)

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


**UPDATE:**

In the bag-level evaluation, our training and evaluation code and the benchmark datasets are mainly downloaded from different online resources. We recently receive a notice that the data input we used during train (as provided in the `datasets/Benchmark` directory) contain an extra feature dimension (e.g., MUSK1 dataset with `x_train`=167 dimensions = 166 +1(instance label), which is placed in the raw data)).  This specific extra dimension literally contains the instance label(not bag label). We unintentionally overlooked this point, which may be because the performance difference caused by the extra dimension in the original `x_train` during our preliminary model validation on MUSK1 was quite minimal (even with very careful tuning(96.8%), the improvement was less than 3% (compared with 94.0% replicated without extra dimension below)), leading us to simply attribute it to an improvement in model capability. Therefore, we continued to use this scheme for other benchmark datasets in the remaining experiments. While the instance label dimension may be open to pick to check some more explicit ability, it is more recommended to re-implement **ALL** methods using the unified version of the data with consistent dimensions, since the additional dimension may bring significant improvements if your model could learn the relation between instance&bag here. But during research, we also validated RGMIL in other bag-level tasks which do not involve feature inconsistency, such as SIVAL image classification, and found that the performance is comparable rather than far more superior to the sota models on bag level. We concluded that RGMIL performs better or comparable to the current mainstream sota models for bag level tasks. This general conclusion could be still verified through the other bag-level tasks and the replication results, see below. 

**REPLICATED UNIFIED VERSION WITHOUT INSTANCE LABEL:**

According to [TRMIL](https://arxiv.org/abs/2307.14025), the +1 feature dimension(instance label) and model capacity of RGMIL may both contribute to the huge improvements.  If you use a version of your data with consistent lower dimension, performance of RGMIL may still remain at sota level generally, but it could decrease. A replicated result of lower dimension by [TRMIL](https://arxiv.org/abs/2307.14025) is as follows:

![image](https://github.com/user-attachments/assets/44f6a61b-bd1c-43a5-803e-7549b6360fe8)

This replication result reflects similar model capacity close to the other experiments we presented and could be considered as a reference. But still, incorporating the exploration of the relationship between instance labels and bag might also be an option in some cases.


 
## MNIST Experiments

Before experiments, use MMNIST/make_ds.py to produce MMNIST datasets first.

You can put MNIST raw dataset under directory "datasets" first.

```python make_ds.py```

It may cost several hours to produce MMNIST.

Run following command to inspect the accuracy float:

```python -m visdom.server```

Run following command to test different aggregators on MMNIST of different modes：

```python main.py --pooling rgp --bagsize 64```

When bagsize set to 1, it corresponds to fully supervised. Choose your options as supported.



## UNBC Experiments
Make sure there are 800GB free disk space for UNBC storage.

Before test on UNBC, put the UNBC original dataset under datasets/UNBC.

Run this command to test RGMIL on UNBC:

```python train_UNBC.py --pooling rgp --dataset UNBC```

It would cost a long time since dataset would be divided and recreated.




