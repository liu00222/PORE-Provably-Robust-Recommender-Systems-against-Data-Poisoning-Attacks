# [PORE: Provably Robust Recommender Systems against Data Poisoning Attacks](https://arxiv.org/pdf/2303.14601.pdf)


This repository contains the code of [PORE](https://arxiv.org/pdf/2303.14601.pdf), which injects backdoors into a pre-trained image encoder such that the downstream classifiers built based on the backdoored image encoder for different downstream tasks simultaneously inherit the backdoor behavior. 

## Citation

If you use this code, please cite the following [paper](https://arxiv.org/pdf/2303.14601.pdf):
```
@inproceedings{jia2023pore,
  title={{PORE}: Provably Robust Recommender Systems against Data Poisoning Attacks},
  author={Jinyuan Jia and Yupei Liu and Yuepeng Hu and Neil Zhenqiang Gong},
  booktitle={USENIX Security Symposium},
  year={2023}
}
```

## Sampling users and making recommendations with base recommenders

The file sample_T.py is used to sample users and make recommendations to them with base recommenders. 

Firstly, you can modify the script run_sample.py to specify the configuration you want. Then, you could run the following script and the results will be saved in ./result: 

```
python3 run_sample.py
```

##### CAVEAT

We load the data in an on-the-fly manner and the experimental results in the paper are all based on the MovieLens datasets that were accessed in 2020.09. However, MovieLens has updated their datasets for several times. Therefore, the results you obtain may be slightly different. To reproduce the results in the paper, please skip this step and directly run the next step with the data that are already in ./data. 

## PORE

The file pore.py implements our PORE. 

You can use the following example script to get the certificatiion results:

```
python3 pore.py
```