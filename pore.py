# set the environment path to find Recommenders
from __future__ import print_function
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
from scipy import stats

from tqdm import tqdm
import statistics
import math
import argparse
from datetime import datetime
import os
import operator as op
from functools import reduce

from util_theorem1 import proportion_confint_calibration, proportion_confint, multi_ci, get_ground_truth, get_bounds, sorted_bounds
from util_theorem1 import get_I_mu, get_ranking,get_C_mu,sum_upper_bounds_in_C_mu,comb,get_comb_n_s_over_comb_n_prime_s,read_test, my_round, drop, process_point,read_frequency_aggregation
from util_theorem1 import construct_bound_dictionary, compute_bound_for_user
from util_sample import preprocess_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='100k')
    parser.add_argument('--s',type=int,default=200)
    parser.add_argument('--N_prime',type=int,default=1)
    parser.add_argument('--T',type=int,default=100000)
    parser.add_argument('--Alg',type=str,default='ir')
    parser.add_argument('--N',type=int,default=10)
    parser.add_argument('--alpha',type=str,default='0.001')
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
    N = args.N
    alpha = float(args.alpha)

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

    # read the saved data
    frequency_aggregation = read_frequency_aggregation(MOVIELENS_DATA_SIZE, s, k, T, args.Alg, 'default', 1)
    test = read_test(MOVIELENS_DATA_SIZE, s, k, T, args.Alg, 'default', 1)

    precision_for_plot = []
    recall_for_plot = []
    e_list = list(range(0, 51))
    n = user_num
    V = range(1, item_num + 1)

    certified_precision_array = np.zeros([user_num, len(e_list)],dtype=np.double)
    certified_recall_array = np.zeros([user_num, len(e_list)],dtype=np.double)

    print("bound dictionary construction starts...")
    bound_dictionary = construct_bound_dictionary(T, user_num, item_num, alpha)
    print("bound dictionary construction ends...")

    combination_table = []
    for e_value in e_list:
        combination_table.append(get_comb_n_s_over_comb_n_prime_s(n, e_value, s))

    for user in range(1, user_num + 1):
        # ground truth
        print(user)
        start = datetime.now()
        I_u = get_ground_truth(test, user)
        lower_bounds, upper_bounds, combined_bounds = compute_bound_for_user(user, frequency_aggregation, item_num, bound_dictionary, I_u)

        # dictionaries: {itemID: lower_bound/upper_bound}
        lower_bounds, upper_bounds, combined_bounds = sorted_bounds(lower_bounds, upper_bounds, combined_bounds)

        t = len(I_u)

        for e_count in range(len(e_list)):
            pi = 0
            e = e_list[e_count]
            n_prime = n + e

            for pi_prime in range(1, t + 1):
                current_mu = list(lower_bounds.keys())[pi_prime - 1]

                if get_ranking(current_mu, combined_bounds) > N:
                    break

                comb_n_s_over_comb_n_prime_s = combination_table[int(e)]
                lhs = lower_bounds[current_mu] * comb_n_s_over_comb_n_prime_s

                C_mu_length = N - get_ranking(current_mu, combined_bounds) + 1
                C_mu = get_C_mu(V, I_u, get_I_mu(V, I_u, current_mu, combined_bounds), upper_bounds, C_mu_length)

                rhs_1 = sum_upper_bounds_in_C_mu(C_mu, upper_bounds) * comb_n_s_over_comb_n_prime_s
                rhs_1 += k * (s/n_prime - (s/n) * comb_n_s_over_comb_n_prime_s)
                rhs_1 = rhs_1 / C_mu_length

                rhs_2 = math.inf
                for v in C_mu:
                    temp = upper_bounds[v] * comb_n_s_over_comb_n_prime_s
                    temp += s / n_prime
                    temp -= (s / n) * comb_n_s_over_comb_n_prime_s
                    rhs_2 = min(rhs_2, temp)

                rhs = min(rhs_1, rhs_2)

                if lhs <= rhs:
                    break

                pi = pi_prime
            certified_precision_array[user-1, e_count] = pi / float(N)
            certified_recall_array[user-1, e_count] = pi / float(t)

    for e_count in range(len(e_list)):
        precision_for_plot.append(np.mean(certified_precision_array[:,e_count]))
        recall_for_plot.append(np.mean(certified_recall_array[:,e_count]))

    print(precision_for_plot)
    print(recall_for_plot)

    # store the result
    result_directory = './results/'+MOVIELENS_DATA_SIZE
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    filename = result_directory + '/'  + "_" + str(s) + "_" + str(k) + "_" + str(T) + "_" + str(
        N) + "_" + str(alpha) + "_"+args.Alg+"_default.csv"

    pd.DataFrame({"e": e_list,
                    "certified_precisions": precision_for_plot,
                    "certified_recalls": recall_for_plot}
                    ).to_csv(filename, index=False)
