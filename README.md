# Active Learning in Continual Learning Scenario

Python implementations of the following active learning algorithms:

- Random Sampling
- Least Confidence
- Margin Sampling
- Entropy Sampling
- Uncertainty Sampling with Dropout Estimation
- Bayesian Active Learning Disagreement
- Cluster-Based Selection
- Adversarial margin

## Prerequisites, download the req.txt file.
- numpy            1.21.2
- scipy            1.7.1
- pytorch          1.10.0
- torchvision      0.11.1
- scikit-learn     1.0.1
- tqdm             4.62.3
- ipdb             0.13.9

You can install the conda environment using :

```
conda env create -f environment.yml
```
OR To install the requirements directly:
```
pip3 install -r req.txt
```

## Demo 
```
  python demo.py \
      --n_round 10 \
      --n_query 2000 \
      --n_init_labeled 10000 \
      --dataset_name CIFAR100 \
      --strategy_name EntropySampling \
      --seed 1
```

The base code of active learning implementation is from the below paper.
```
@article{Huang2021deepal,
    author    = {Kuan-Hao Huang},
    title     = {DeepAL: Deep Active Learning in Python},
    journal   = {arXiv preprint arXiv:2111.15258},
    year      = {2021},
}
```
