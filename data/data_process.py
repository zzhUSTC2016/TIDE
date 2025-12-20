# coding=utf-8
import os
import math
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

sys.path.append('../')
from config import opt


def read_ratings(file_path, sep=','):
    """
    自动判断文件是否有表头，并读取 CSV/TSV 文件
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip().split(sep)
    if first_line[0].lower() == 'userid':
        header = 0
    else:
        header = None

    df = pd.read_csv(file_path, sep=sep, header=header)
    df.columns = ['user', 'item', 'rating', 'timestamp']
    return df


def process(dataset):
    print("1  filtering users and items")

    # 选择文件路径
    data_dir = 'data/' + dataset
    csv_path = os.path.join(data_dir, 'ratings.csv')
    tsv_path = os.path.join(data_dir, 'ratings.tsv')
    if os.path.exists(csv_path):
        data_path = csv_path
        sep = ','
    elif os.path.exists(tsv_path):
        data_path = tsv_path
        sep = '\t'
    else:
        raise FileNotFoundError("Cannot find ratings.csv or ratings.tsv in dataset folder.")

    # 读取文件
    all_data = read_ratings(data_path, sep)

    # 确保列类型正确
    all_data['user'] = all_data['user'].astype(str)
    all_data['item'] = all_data['item'].astype(str)
    all_data['rating'] = pd.to_numeric(all_data['rating'], errors='coerce')
    all_data['timestamp'] = pd.to_numeric(all_data['timestamp'], errors='coerce')
    all_data = all_data.dropna(subset=['rating', 'timestamp']).reset_index(drop=True)

    dt2010 = datetime(2010, 1, 1)
    timestamp2010 = dt2010.timestamp()

    # 过滤 Douban-movie 时间
    if opt.dataset == 'Douban-movie':
        data_filtered = all_data[all_data['timestamp'] > timestamp2010].reset_index(drop=True)
    else:
        data_filtered = all_data.copy()

    # 使用 groupby 完全向量化构建 user_dict 和 item_dict
    user_dict = data_filtered.groupby('user')['item'].agg(list).to_dict()
    item_dict = data_filtered.groupby('item')['user'].agg(list).to_dict()

    # 用户/物品过滤
    f_user_dict, f_item_dict = {}, {}
    n_u_f, n_i_f = opt.n_u_f, opt.n_i_f
    print('n_users\tn_items')
    while True:
        print(len(user_dict.keys()), len(item_dict.keys()))
        flag1, flag2 = True, True

        # 用户过滤
        for u_id in user_dict.keys():
            pos_items = user_dict[u_id]
            val_items = [idx for idx in pos_items if idx in item_dict.keys()]
            if len(val_items) >= n_u_f:
                f_user_dict[u_id] = val_items
            else:
                flag1 = False
        user_dict = f_user_dict.copy()

        # 物品过滤
        for i_id in item_dict.keys():
            pos_users = item_dict[i_id]
            val_users = [udx for udx in pos_users if udx in user_dict.keys()]
            if len(pos_users) >= n_i_f:
                f_item_dict[i_id] = val_users
            else:
                flag2 = False
        item_dict = f_item_dict.copy()
        f_user_dict, f_item_dict = {}, {}

        if flag1 and flag2:
            print('filter done.')
            break

    # 过滤 DataFrame
    interactions_num_1 = len(all_data)
    all_data = all_data[all_data['user'].isin(user_dict.keys())].reset_index(drop=True)
    all_data = all_data[all_data['item'].isin(item_dict.keys())].reset_index(drop=True)
    if opt.dataset == 'Douban-movie':
        all_data = all_data[all_data['timestamp'] > timestamp2010].reset_index(drop=True)
    interactions_num_2 = len(all_data)
    print('interactions_num:\n', interactions_num_1, interactions_num_2)

    # ID remap
    le = LabelEncoder()
    all_data['user'] = le.fit_transform(all_data['user'])
    all_data['item'] = le.fit_transform(all_data['item'])
    all_data = all_data.sort_values(by=['user', 'item']).reset_index(drop=True)

    user_num = all_data['user'].max() + 1
    item_num = all_data['item'].max() + 1

    min_time = all_data['timestamp'].min()
    max_time = all_data['timestamp'].max()

    # PDA流行度
    K = 10
    time_interval = math.ceil((max_time - min_time) / K)
    PDA_popularity = pd.DataFrame(index=np.arange(item_num))
    data_concat = pd.DataFrame()
    for i in range(K):
        time_split_min = min_time + time_interval * i
        time_split_max = time_split_min + time_interval
        if i == K - 1:
            time_split_max = all_data['timestamp'].max() + 1
        data_split = all_data[
            (all_data['timestamp'] >= time_split_min) & (all_data['timestamp'] < time_split_max)].reset_index(drop=True)
        count = data_split.item.value_counts()
        pop = pd.DataFrame(count)
        pop.columns = [str(i)]
        pop = pop / pop.values.max()
        PDA_popularity = pd.merge(PDA_popularity, pop, left_index=True, right_index=True, how='left').fillna(0)
        data_split['split_idx'] = i
        data_concat = pd.concat([data_concat, data_split]).reset_index(drop=True)
    all_data = data_concat
    PDA_popularity.to_csv(os.path.join(data_dir, 'PDA_popularity.csv'), index=0, sep='\t')

    # DICE、IPS流行度
    count = all_data.item.value_counts()
    pop = count / count.max()
    np.save(opt.DICE_popularity_path, pop)

    # 划分训练/验证/测试集
    print("4  split training set and test set")
    all_data = all_data.sort_values(by=['timestamp']).reset_index(drop=True)
    train_time_max = min_time + time_interval * (K - 1)
    train_data = all_data[all_data['timestamp'] < train_time_max]
    rest_data = all_data[all_data['timestamp'] >= train_time_max].reset_index(drop=True)
    val_user_set = np.random.choice(np.arange(user_num), int(user_num / 2), replace=False)
    val_data = rest_data[rest_data['user'].isin(val_user_set)].reset_index(drop=True)
    test_data = rest_data[~rest_data['user'].isin(val_user_set)].reset_index(drop=True)

    test_data_high_rating = test_data.loc[test_data['rating'] > 4].reset_index(drop=True)
    val_data_high_rating = val_data.loc[val_data['rating'] > 4].reset_index(drop=True)

    # 保存 CSV 文件
    all_data.to_csv(os.path.join(data_dir, 'all_data.csv'), index=0, sep='\t')
    train_data.to_csv(os.path.join(data_dir, 'train_data.csv'), index=0, sep='\t')
    val_data.to_csv(os.path.join(data_dir, 'val_data.csv'), index=0, sep='\t')
    test_data.to_csv(os.path.join(data_dir, 'test_data.csv'), index=0, sep='\t')

    # 保存 train_list.txt
    train_data = train_data.sort_values(by=['user']).reset_index(drop=True)
    train_user_count = train_data.groupby('user')['user'].count()
    f_train = open(os.path.join(data_dir, 'train_list.txt'), 'w')
    u = train_data['user'][0]
    user_interaction_num = train_user_count[u]
    f_train.write(str(u) + '\t' + str(user_interaction_num) + '\t' + str(train_data['item'][0]))
    for i in range(1, train_data.shape[0]):
        if train_data['user'][i] == train_data['user'][i - 1]:
            f_train.write('\t' + str(train_data['item'][i]))
        else:
            u = train_data['user'][i]
            user_interaction_num = train_user_count[u]
            f_train.write('\n' + str(u) + '\t' + str(user_interaction_num) + '\t' + str(train_data['item'][i]))
    f_train.close()

    # 保存测试/验证列表
    def test_process(test_data, test_data_high_rating, val=True):
        test_data = test_data.sort_values(by=['user']).reset_index(drop=True)
        if val:
            f_test = open(os.path.join(data_dir, 'val_list.txt'), 'w')
        else:
            f_test = open(os.path.join(data_dir, 'test_list.txt'), 'w')
        f_test.write(str(test_data['user'][0]) + '\t' + str(test_data['item'][0]))
        for i in range(1, test_data.shape[0]):
            if test_data['user'][i] == test_data['user'][i - 1]:
                f_test.write('\t' + str(test_data['item'][i]))
            else:
                f_test.write('\n' + str(test_data['user'][i]) + '\t' + str(test_data['item'][i]))
        f_test.close()

        test_data_high_rating = test_data_high_rating.sort_values(by=['user']).reset_index(drop=True)
        if val:
            f_test = open(os.path.join(data_dir, 'val_high_rating_list.txt'), 'w')
        else:
            f_test = open(os.path.join(data_dir, 'test_high_rating_list.txt'), 'w')
        f_test.write(str(test_data_high_rating['user'][0]) + '\t' + str(test_data_high_rating['item'][0]))
        for i in range(1, test_data_high_rating.shape[0]):
            if test_data_high_rating['user'][i] == test_data_high_rating['user'][i - 1]:
                f_test.write('\t' + str(test_data_high_rating['item'][i]))
            else:
                f_test.write('\n' + str(test_data_high_rating['user'][i]) + '\t' + str(test_data_high_rating['item'][i]))
        f_test.close()

        if val:
            f_test_rating = open(os.path.join(data_dir, 'val_list_rating.txt'), 'w')
        else:
            f_test_rating = open(os.path.join(data_dir, 'test_list_rating.txt'), 'w')
        f_test_rating.write(str(test_data['user'][0]) + '\t' + str(test_data['rating'][0]))
        for i in range(1, test_data.shape[0]):
            if test_data['user'][i] == test_data['user'][i - 1]:
                f_test_rating.write('\t' + str(test_data['rating'][i]))
            else:
                f_test_rating.write('\n' + str(test_data['user'][i]) + '\t' + str(test_data['rating'][i]))
        f_test_rating.close()

    test_process(val_data, val_data_high_rating, val=True)
    test_process(test_data, test_data_high_rating, val=False)

    # item_interactions.csv
    print("saving item_interactions.csv")
    train_data = train_data.sort_values(by=['item', 'timestamp']).reset_index(drop=True)
    f_item_interaction = open(os.path.join(data_dir, 'item_interactions.csv'), 'w')
    item_id = train_data['item'][0]
    item_idx = 0
    interaction_num = 1
    str_to_write = str(int(train_data['timestamp'][0]))

    for i in range(1, train_data.shape[0]):
        if train_data['item'][i] == train_data['item'][i - 1]:
            interaction_num += 1
            str_to_write = str_to_write + ',' + str(int(train_data['timestamp'][i]))
        else:
            str_to_write = str(item_id) + ',' + str(interaction_num) + ',' + str_to_write + '\n'
            f_item_interaction.write(str_to_write)
            item_id = train_data['item'][i]
            item_idx += 1
            while item_id != item_idx:  # 有item没有交互
                str_to_write = str(item_idx) + ',' + str(0) + '\n'
                f_item_interaction.write(str_to_write)
                item_idx += 1
            str_to_write = str(int(train_data['timestamp'][i]))
            interaction_num = 1
    str_to_write = str(item_id) + ',' + str(interaction_num) + ',' + str_to_write + '\n'
    f_item_interaction.write(str_to_write)
    item_idx += 1
    while item_num != item_idx:  # 有item没有交互
        str_to_write = str(item_idx) + ',' + str(0) + '\n'
        f_item_interaction.write(str_to_write)
        item_idx += 1
    f_item_interaction.close()
    print("saved!")


def PDA_test_pop(dataset):
    PDA_popularity = pd.read_csv('data/' + dataset + '/PDA_popularity.csv', sep='\t')
    PDA_popularity['9'] = PDA_popularity['8'] + opt.PDA_alpha * (PDA_popularity['8'] - PDA_popularity['7'])
    return PDA_popularity['9'].values
    '''PDA_popularity['9'] = (PDA_popularity['9'] - np.min(PDA_popularity['9'])) / \
                          (np.max(PDA_popularity['9']) - np.min(PDA_popularity['9']))
    PDA_popularity.to_csv('data/' + dataset + '/PDA_popularity.csv', index=0, sep='\t')'''


