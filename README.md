
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

Run following command to test DSMIL on these datasets for comparison：

```python train_mil.py```

You can also modify the **DATASETS** array in train_mil.py to perform specific experiments on different benchmark datasets.


**UPDATE**

In the bag-level evaluation, our training and evaluation code comes from [DSMIL](https://github.com/binli123/dsmil-wsi), and the benchmark datasets are downloaded from online resources. This is the version we used in paper and presented publicly. However, we recently noticed that the version of dataset input we used during train (as provided in the `datasets/Benchmark` directory) contain an extra feature dimension (e.g., FOX dataset with 230+1 dimensions).  This specific extra dimension literally contains the instance label(not bag label). The instance label dimension may be open to pick to check some explicit ability. It is more recommended to re-implement **ALL** methods using the unified version of the dataset with consistent dimensions, since the additional dimension may bring significant improvements if your model could find the relation here. However this dimension consistency is what we did not notice before, and we were supposed to re-implement **ALL** the methods in the datasets(MUSK1,MUSK2,FOX,TIGER,ELEPHANTS), rather than collecting the raw statistics of previous methods. 


According to [TRMIL](https://arxiv.org/abs/2307.14025), the +1 feature dimension(instance label) and model capacity of RGMIL may both contribute to the huge improvements. You may get similar performance as presented in paper if you use our full scheme here, which you will find a huge improvement than previous. While it may approach **SUPER HIGH** performance in such full scheme, if you use a version of your data with lower dimension, performance of RGMIL may still remain at sota level, but it could decrease! A reproduced result of lower dimension by [TRMIL](https://arxiv.org/abs/2307.14025) is as follows:

![image](https://github.com/user-attachments/assets/44f6a61b-bd1c-43a5-803e-7549b6360fe8)


 
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




