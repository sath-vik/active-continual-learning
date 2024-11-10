# DeepAL: Deep Active Learning in Python

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

You can also use the following command to install conda environment

```
conda env create -f environment.yml

pip3 install -r req.txt
```

## Demo 

```
  python demo.py \
      --n_round 10 \
      --n_query 2000 \
      --n_init_labeled 10000 \
      --dataset_name MNIST \
      --strategy_name EntropySampling \
      --seed 1
```

## Citing
The base code of active learning implementation is from the below paper.
```
@article{Huang2021deepal,
    author    = {Kuan-Hao Huang},
    title     = {DeepAL: Deep Active Learning in Python},
    journal   = {arXiv preprint arXiv:2111.15258},
    year      = {2021},
}
```

## Reference

[1] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[2] Active Hidden Markov Models for Information Extraction, IDA, 2001

[3] Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009

[4] Deep Bayesian Active Learning with Image Data, ICML, 2017

[5] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[6] Adversarial Active Learning for Deep Networks: a Margin Based Approach, arXiv, 2018
