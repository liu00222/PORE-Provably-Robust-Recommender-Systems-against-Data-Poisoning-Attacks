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

import statistics
import math

import operator as op
from functools import reduce


def proportion_confint_calibration(alpha_2,upper_bound,count,nobs,re=0.0001):
    sf_alpha_value = stats.beta.sf(upper_bound,count + 1, nobs - count)
    if np.abs((sf_alpha_value-alpha_2)/(alpha_2))>re:
        low,high,mid=0.0,1.0,0.5
        sf_value = stats.beta.sf(mid,count + 1, nobs - count)
        while np.abs((sf_value-alpha_2)/(alpha_2))>re and np.abs(low - high) > 1e-10:
            if sf_value > alpha_2:
                low = mid
            else:
                high = mid
            mid = (high+low)/2.0
            sf_value = stats.beta.sf(mid,count + 1, nobs - count)
        return mid
    else:
        return upper_bound

def proportion_confint(count, nobs, alpha=0.05, method='beta'):
    pd_index = getattr(count, 'index', None)
    if pd_index is not None and callable(pd_index):
        # this rules out lists, lists have an index method
        pd_index = None
    count = np.asarray(count)
    nobs = np.asarray(nobs)
    q_ = count * 1. / nobs
    alpha_2 = 0.5 * alpha
    if method=='beta':
        ci_low = stats.beta.ppf(alpha_2, count, nobs - count + 1)
        ci_upp = stats.beta.isf(alpha_2, count + 1, nobs - count)
        ci_upp = proportion_confint_calibration(alpha_2,ci_upp,count,nobs)

        if np.ndim(ci_low) > 0:
            ci_low[q_ == 0] = 0
            ci_upp[q_ == 1] = 1
        else:
            ci_low = ci_low if (q_ != 0) else 0
            ci_upp = ci_upp if (q_ != 1) else 1
    else:
        raise NotImplementedError
    return ci_low, ci_upp


def multi_ci(counts, user_num, T, alpha=0.001):
    multi_list = []
    n = T
    l = len(counts)
    alpha = alpha / user_num
    for i in range(l):
        multi_list.append(proportion_confint(min(max(counts[i], 1e-10), n-1e-10), n, alpha=alpha*2./l, method="beta"))
    return np.array(multi_list)


def get_ground_truth(test, user):
    return test[test['userID'] == user]['itemID'].tolist()

def construct_bound_dictionary(T, user_num, item_num, alpha):
    multi_list = []
    alpha = alpha / float(user_num * item_num)
    for i in range(T+1):
        multi_list.append(proportion_confint(min(max(i, 1e-10), T-1e-10), T, alpha=alpha*2., method="beta"))
    return np.array(multi_list)
    
def compute_bound_for_user(user, frequency_aggregation, item_num, bound_dictionary, ground_truth):
    # dictionary, e.g. {item_id: frequency}
    freq_of_user = frequency_aggregation[user - 1]

    # frequency of all the items for a specific user
    items_freq_count = [0 for i in range(item_num)]

    for key in freq_of_user.keys():
        items_freq_count[key - 1] = freq_of_user[key]

    bounds = np.zeros([item_num, 2],dtype=np.double)
    for i in range(len(items_freq_count)):
        bounds[i,0] = bound_dictionary[items_freq_count[i],0]
        bounds[i,1] = bound_dictionary[items_freq_count[i],1]

    lower_bounds = {}
    upper_bounds = {}
    combined_bounds = {}

    for itemID in range(1, item_num + 1):
        if itemID in ground_truth:
            lower_bounds[itemID] = bounds[itemID - 1][0]
            combined_bounds[itemID] = bounds[itemID - 1][0]
        else:
            upper_bounds[itemID] = bounds[itemID - 1][1]
            combined_bounds[itemID] = bounds[itemID - 1][1]

    return lower_bounds, upper_bounds, combined_bounds
def get_bounds(user, T, user_num, item_num, frequency_aggregation, ground_truth):
    # dictionary, e.g. {item_id: frequency}
    freq_of_user = frequency_aggregation[user - 1]

    # frequency of all the items for a specific user
    items_freq_count = [0 for i in range(item_num)]

    for key in freq_of_user.keys():
        items_freq_count[key - 1] = freq_of_user[key]

    # bounds = [[lower1, upper1], [lower2, upper2], ...]
    bounds = multi_ci(np.array(items_freq_count), user_num, T)

    # lower_bounds = {item_id: lower_bounds}
    # upper_bounds = {item_id: upper_bounds}
    lower_bounds = {}
    upper_bounds = {}
    combined_bounds = {}

    for itemID in range(1, item_num + 1):
        if itemID in ground_truth:
            lower_bounds[itemID] = bounds[itemID - 1][0]
            combined_bounds[itemID] = bounds[itemID - 1][0]
        else:
            upper_bounds[itemID] = bounds[itemID - 1][1]
            combined_bounds[itemID] = bounds[itemID - 1][1]

    return lower_bounds, upper_bounds, combined_bounds


def sorted_bounds(lower_bounds, upper_bounds, combined_bounds):
    # [(itemID, bounds), ...]
    lower_bounds_sorted_list = sorted(lower_bounds.items(), key=lambda x:x[1], reverse=True)
    upper_bounds_sorted_list = sorted(upper_bounds.items(), key=lambda x:x[1], reverse=True)
    combined_bounds_sorted_list = sorted(combined_bounds.items(), key=lambda x:x[1], reverse=True)

    lower_bounds_sorted_dict = {}
    upper_bounds_sorted_dict = {}
    combined_bounds_sorted_dict = {}

    for element in lower_bounds_sorted_list:
        lower_bounds_sorted_dict[element[0]] = element[1]

    for element in upper_bounds_sorted_list:
        upper_bounds_sorted_dict[element[0]] = element[1]

    for element in combined_bounds_sorted_list:
        combined_bounds_sorted_dict[element[0]] = element[1]

    return lower_bounds_sorted_dict, upper_bounds_sorted_dict, combined_bounds_sorted_dict


def get_I_mu(V, I_u, mu, combined_bounds):
    """
    Get the set I_mu as defined in the theorem 1

    :param V: <list> a list of all item IDs
    :param I_u: <list> ground truth
    :param mu: <int> item ID of an item in the ground truth
    :return: <list> I_mu
    """
    # get V - I_u
    V_minus_I_u = []
    for v in V:
        if v not in I_u:
            V_minus_I_u.append(v)

    # get items in V - I_u whose ranking is higher than ranking_mu
    ranking_mu = get_ranking(mu, combined_bounds)
    I_mu = []
    for v in V_minus_I_u:
        if get_ranking(v, combined_bounds) < ranking_mu:
            I_mu.append(v)

    return I_mu


def get_ranking(mu, combined_bounds):
    """
    Get the ranking of an item mu in the combined set

    :param mu: <int> item ID
    :return: <int> the ranking which is from 1 to item_num
    """
    return list(combined_bounds.keys()).index(mu) + 1


def get_C_mu(V, I_u, I_mu, upper_bounds, num):
    C_mu = []
    current_num = 0
    for key in upper_bounds.keys():
        if current_num >= num:
            break
        else:
            if key not in I_u and key not in I_mu:
                C_mu.append(key)
                current_num += 1
    return C_mu


def sum_upper_bounds_in_C_mu(C_mu, upper_bounds):
    result = 0
    for i in C_mu:
        result += upper_bounds[i]
    return result


def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def get_comb_n_s_over_comb_n_prime_s(n, e, s):
    result = 1
    for i in range(1, e + 1):
        result = result * (n + i - s)
        result = result / (n + i)
    return result




def read_test(dataset, s, k, T, method, alpha, count):
    file = './data/'+dataset + '/'+ dataset + '_' + str(s) + '_' + str(k) + '_' + str(T) +  '_' + method + '_' + alpha + '/'
    file += 'test_' + str(count) + '.csv'

    print(file)

    # top_N, test
    return pd.read_csv(file)


def my_round(num):
    if num - int(num) >= 0.5:
        return int(num + 1)
    return int(num)


def drop(text, char_lst):
    buffer = ""
    for i in range(len(text)):
        if text[i] not in char_lst:
            buffer += text[i]
    return buffer


def process_point(text):
    text = drop(text, ['[', ']', '(', ')', '\n'])
    raw = text.split(';')
    points = {}
    for i in range(len(raw)):
        if raw[i] != '':
            point = raw[i].split(',')
            points[int(point[0])] = int(point[1])
    return points


def read_frequency_aggregation(dataset, s, k, T, method, alpha, count):
    file = './data/'+ dataset + '/' + dataset + '_' + str(s) + '_' + str(k) + '_' + str(T)  + '_' + method + '_' + alpha + '/'
    file += 'frequency_aggregation_' + str(count) + '.txt'

    print(file)

    with open(file) as f:
        temp = f.readlines()
    frequency_aggregation = []
    for i in range(len(temp)):
        frequency_aggregation.append(process_point(temp[i]))

    return frequency_aggregation
