# set the environment path to find Recommenders
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from reco_utils.common.python_utils import binarize
from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from reco_utils.recommender.sar import SAR
import os
from tqdm import tqdm
import argparse

from util_sample import split_by_user, train_model, sampling, predict, save_aggregation, save_test, load_data_after_split, save_data_after_split
from util_sample import preprocess_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='100k')
    parser.add_argument('--s',type=int,default=300)
    parser.add_argument('--N_prime',type=int,default=1)
    parser.add_argument('--T',type=int,default=100000)
    parser.add_argument('--Alg',type=str,default='ir')
    parser.add_argument('--series',type=int,default=1)
    args = preprocess_args(parser.parse_args())

    # Select MovieLens data size: 100k, 1m, 10m, or 20m
    MOVIELENS_DATA_SIZE = args.dataset

    data = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE
    )

    # Convert the float precision to 32-bit in order to reduce memory consumption
    data['rating'] = data['rating'].astype(np.float32)

    s = args.s
    k = args.k
    T = args.T
    series = args.series

    user_num = 0
    item_num = 0

    if MOVIELENS_DATA_SIZE == '1m':
        user_num = 6040
        item_num = 3952
    elif MOVIELENS_DATA_SIZE == '100k':
        user_num = 943
        item_num = 1682
    else:
        raise ValueError

    data_after_split = split_by_user(data, "userID", user_num, args.Alg)
    print("Complete splitting the dataset. ")

    frequency_aggregation = [{} for i in range(user_num)]

    for i in tqdm(range(T)):
        # randomly sample s users and get the train set and test set
        train_set, test_set = sampling(data_after_split, user_num, s)
        model = train_model(args.Alg, train_set)
        top_k = predict(model, test_set, train_set, k, args.Alg, user_num)

        # store the recommended items into frequency_aggregation
        for j in range(len(top_k)):
            # remember user position = user id - 1
            current_user = top_k['userID'][j]
            current_item = top_k['itemID'][j]
            if current_item in frequency_aggregation[current_user - 1]:
                frequency_aggregation[current_user - 1][current_item] += 1
            else:
                frequency_aggregation[current_user - 1][current_item] = 1

    test = data_after_split[0]['test']
    for i in tqdm(range(1, user_num)):
        test = test.append(data_after_split[i]['test'], ignore_index=True)

    directory = './data/' + args.dataset
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory += '/' + MOVIELENS_DATA_SIZE + '_' + str(s) + '_' + str(k) + '_' + str(T) + '_' + args.Alg + '_' + 'default'
    if not os.path.exists(directory):
        os.makedirs(directory)

    test_name = directory + '/test_' + str(series) + '.csv'
    save_test(test, test_name)

    aggregation_name = directory + '/frequency_aggregation_' + str(series) + '.txt'
    save_aggregation(frequency_aggregation, aggregation_name)

    print('Complete')
