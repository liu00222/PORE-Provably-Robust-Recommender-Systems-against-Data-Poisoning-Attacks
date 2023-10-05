# set the environment path to find Recommenders
import sys
import cornac
import papermill as pm
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from reco_utils.common.python_utils import binarize
from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.cornac.cornac_utils import predict_ranking
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
from reco_utils.common.constants import (
    COL_DICT,
    DEFAULT_K,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_TIMESTAMP_COL,
    SEED,
)
from reco_utils.recommender.sar import SAR
import os


# Utils and constants for BPR
EPOCHS = 15


def prepare_training_bpr(train):
    return cornac.data.Dataset.from_uir(
        train.itertuples(index=False), seed=SEED
    )


def train_bpr(params, data):
    model = cornac.models.BPR(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def prepare_metrics_fastai(train, test):
    data = test.copy()
    data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].astype("str")
    data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].astype("str")
    return train, data


def recommend_k_bpr(model, test, train):
    with Timer() as t:
        topk_scores = predict_ranking(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=True,
        )
    return topk_scores, t


def ranking_metrics_python(test, predictions, k=DEFAULT_K):
    return {
        "MAP": map_at_k(test, predictions, k=k, **COL_DICT),
        "nDCG@k": ndcg_at_k(test, predictions, k=k, **COL_DICT),
        "Precision@k": precision_at_k(test, predictions, k=k, **COL_DICT),
        "Recall@k": recall_at_k(test, predictions, k=k, **COL_DICT),
    }


prepare_training_data = {
    "bpr": prepare_training_bpr,
}

bpr_params = {
    "k": 200,
    "max_iter": EPOCHS,
    "learning_rate": 0.075,
    "lambda_reg": 1e-3,
    "seed": SEED,
    "verbose": False
}

params = {
    "bpr": bpr_params,
}

trainer = {
    "bpr": lambda params, data: train_bpr(params, data),
}

prepare_metrics_data = {
    "fastai": lambda train, test: prepare_metrics_fastai(train, test),
}

ranking_predictor = {
    "bpr": lambda model, test, train: recommend_k_bpr(model, test, train),
}

ranking_evaluator = {
    "bpr": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
}


# Assumption: users = [1, 2, 3, ..., user_num]
def split_by_user(data, col_user, user_num, algorithm):
    """
    For each user (among n users), we can split it rating scores as training and testing,
    e.g., 75% for training and 25% for testing.

    Currently with the assumption that users_id = [1, 2, 3, ..., user_num]

    :param data: <pd.frame.DataFrame> the data set
    :param col_user: <string> the column name of the users
    :param user_num: <int> number of users
    :param algorithm: <string> sar or bpr
    :return:
    """
    from tqdm import tqdm
    if algorithm == 'sar':
        data_after_split = []
        for i in tqdm(range(user_num)):
            user_temp = data[data[col_user] == (i + 1)]

            # Use the same setting as Microsoft did to split the train and test
            train, test = python_stratified_split(user_temp, ratio=0.75, col_user='userID', col_item='itemID', seed=42)
            data_after_split.append({"train": train, "test": test})

        # Note that the user id at ith position is i + 1, e.g. the user id of data_after_split[0] is 1
        return data_after_split
    elif algorithm == 'bpr':
        train, test = python_stratified_split(data,
                                                    ratio=0.75,
                                                    min_rating=1,
                                                    filter_by="item",
                                                    col_user=DEFAULT_USER_COL,
                                                    col_item=DEFAULT_ITEM_COL
                                                    )
        data_after_split = []
        for user in range(1, user_num + 1):
            current_train = train[train['userID'] == user]
            current_test = test[test['userID'] == user]
            data_after_split.append({"train": current_train, "test": current_test})
        return data_after_split
    else:
        raise NotImplementedError


def save_data_after_split(dataset_directory, data_after_split, user_num):
    test = data_after_split[0]['test']
    for i in range(1, user_num):
        test = test.append(data_after_split[i]['test'], ignore_index=True)

    test.to_csv(dataset_directory + '/test.csv', index=False)

    train = data_after_split[0]['train']
    for i in range(1, user_num):
        train = train.append(data_after_split[i]['train'], ignore_index=True)

    train.to_csv(dataset_directory + '/train.csv', index=False)


def load_data_after_split(dataset_directory, user_num):
    test = pd.read_csv(dataset_directory + '/test.csv')
    train = pd.read_csv(dataset_directory + '/train.csv')

    data_after_split = []

    for i in range(user_num):
        data_after_split.append({'train': train[train['userID'] == (i+1)], 'test': test[test['userID'] == (i+1)]})

    return data_after_split


def sampling(data_after_split, user_num, s):
    # sampled_users_indices contains the sampled user index in data_after_split, not the user id
    sampled_users_indices = np.random.choice(a=range(user_num), size=s, replace=False)

    train_set = data_after_split[sampled_users_indices[0]]['train']
    test_set = data_after_split[sampled_users_indices[0]]['test']

    for index in sampled_users_indices:
        if index != sampled_users_indices[0]:
            train_set = train_set.append(data_after_split[index]['train'], ignore_index=True)
            test_set = test_set.append(data_after_split[index]['test'], ignore_index=True)

    return train_set, test_set


def train_model(algorithm, train_set):
    if algorithm == 'sar':
        model = SAR(
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_timestamp="timestamp",
            similarity_type="jaccard",
            time_decay_coefficient=30,
            timedecay_formula=True
        )
        model.fit(train_set)

        return model
    elif algorithm == 'bpr':
        train = prepare_training_data.get(algorithm, lambda x: x)(train_set)
        model_params = params[algorithm]
        model, time_train = trainer[algorithm](model_params, train)
        return model
    else:
        raise NotImplementedError


def predict(model, test_set, train_set, k, algorithm, user_num):
    if algorithm == 'sar':
        top_k = model.recommend_k_items(test_set, top_k=k, remove_seen=True)
        return top_k
    elif algorithm == 'bpr':
        top_k_scores, time_ranking = ranking_predictor[algorithm](model, test_set, train_set)
        top_k = get_k_recommended_items(top_k_scores, k, user_num)
        return top_k
    else:
        raise NotImplementedError


def get_k_recommended_items(top_k_scores, k, user_num):
    result_user = []
    result_item = []

    for userID in range(1, user_num+1):
        if not top_k_scores[top_k_scores['userID'] == userID].empty:
            itemIDs = list(top_k_scores[top_k_scores['userID'] == userID].sort_values('prediction')[-k:]['itemID'])
            for itemID in itemIDs:
                result_user.append(userID)
                result_item.append(itemID)

    return pd.DataFrame({'userID': result_user, 'itemID': result_item})


def preprocess_args(args):
    # 'ir' alg. in the paper is the 'sar' alg. in Microsoft's library
    if args.Alg == 'ir':
        args.Alg = 'sar'
        
    # k is the N' in the paper
    args.k = args.N_prime
    return args

def save_test(test,filename):
    test.to_csv(filename, index=False)
    return


def save_aggregation(frequency_aggregation,filename):
    print("[", end="", file=open(filename, "a"))
    for i in range(len(frequency_aggregation)):
        current_dict = frequency_aggregation[i]
        for j in range(len(list(current_dict.keys()))):
            key = list(current_dict.keys())[j]
            p = [key, current_dict[key]]
            print("(" + str(p[0]) + "," + str(p[1]) + ")", end="", file=open(filename, "a"))
            if j != len(list(current_dict.keys())) - 1:
                print(";", end="", file=open(filename, "a")),
        print("]", file=open(filename, "a"))
    return
