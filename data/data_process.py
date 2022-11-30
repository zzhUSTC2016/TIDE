# coding=utf-8
# 读取数据，过滤出有效的user和item列表
import numpy as np
import pandas as pd
import math
import sys
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

sys.path.append('../')
from config import opt


def process(dataset):
    print("1  filtering users and items")
    data_path = 'data/' + dataset + '/ratings.csv'

    f = open(data_path, 'r')
    lines = f.readlines()
    user_dict, item_dict = {}, {}
    dt2010 = datetime(2010, 1, 1)  # 用指定日期时间创建datetime
    timestamp2010 = dt2010.timestamp()  # 把datetime转换为timestamp
    for line in lines:
        temp = line.strip().split(',')
        u_id = temp[0]
        i_id = temp[1]
        timestamp = float(temp[-1])
        if opt.dataset == 'Douban-movie':
            if timestamp > timestamp2010:
                if u_id in user_dict.keys():
                    user_dict[u_id].append(i_id)
                else:
                    user_dict[u_id] = [i_id]

                if i_id in item_dict.keys():
                    item_dict[i_id].append(u_id)
                else:
                    item_dict[i_id] = [u_id]
        else:
            if u_id in user_dict.keys():
                user_dict[u_id].append(i_id)
            else:
                user_dict[u_id] = [i_id]

            if i_id in item_dict.keys():
                item_dict[i_id].append(u_id)
            else:
                item_dict[i_id] = [u_id]

    f_user_dict, f_item_dict = {}, {}

    n_u_f, n_i_f = opt.n_u_f, opt.n_i_f
    print('n_users\tn_items')
    while True:
        print(len(user_dict.keys()), len(item_dict.keys()))
        flag1, flag2 = True, True

        for u_id in user_dict.keys():
            pos_items = user_dict[u_id]
            val_items = [idx for idx in pos_items if idx in item_dict.keys()]

            if len(val_items) >= n_u_f:
                f_user_dict[u_id] = val_items
            else:
                flag1 = False

        user_dict = f_user_dict.copy()

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
    
    # 过滤
    all_data = pd.read_csv(data_path, sep=',', header=None, names=['user', 'item', 'rating', 'timestamp'], dtype={'user':object, 'item':object})
    interactions_num_1 = len(all_data)
    all_data = all_data[all_data['user'].isin(user_dict.keys())].reset_index(drop=True)
    all_data = all_data[all_data['item'].isin(item_dict.keys())].reset_index(drop=True)
    if opt.dataset == 'Douban-movie':
        all_data = all_data[all_data['timestamp'] > timestamp2010].reset_index(drop=True)
    interactions_num_2 = len(all_data)
    print('interactions_num:\n', interactions_num_1, interactions_num_2)

    # ID remap
    le = LabelEncoder()
    le.fit(all_data['user'].values)
    all_data['user'] = le.transform(all_data['user'].values)
    le.fit(all_data['item'].values)
    all_data['item'] = le.transform(all_data['item'].values)
    all_data = all_data.sort_values(by=['user', 'item']).reset_index(drop=True)
    # print('after ID remap:\n',all_data)

    user_num = all_data['user'].max() + 1
    item_num = all_data['item'].max() + 1

    min_time = all_data['timestamp'].min()
    max_time = all_data['timestamp'].max()

    # 计算PDA流行度
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
        # print(time_split_min, time_split_max, len(data_split))
        count = data_split.item.value_counts()
        pop = pd.DataFrame(count)
        pop.columns = [str(i)]
        # (pop, pop.values.max())
        pop = pop / pop.values.max()
        PDA_popularity = pd.merge(PDA_popularity, pop, left_index=True, right_index=True, how='left').fillna(0)
        data_split['split_idx'] = i
        data_concat = pd.concat([data_concat, data_split]).reset_index(drop=True)
    all_data = data_concat
    PDA_popularity.to_csv('data/' + dataset + '/PDA_popularity.csv', index=0, sep='\t')

    # 计算DICE、IPS流行度
    count = all_data.item.value_counts()
    pop = count / count.max()
    np.save(opt.DICE_popularity_path, pop)

    # 划分训练集，验证集和测试集
    print("4  split training set and test set")
    all_data = all_data.sort_values(by=['timestamp']).reset_index(drop=True)
    time_interval = math.ceil((max_time - min_time) / K)
    train_time_max = min_time + time_interval * (K - 1)
    train_data = all_data[all_data['timestamp'] < train_time_max]
    rest_data = all_data[all_data['timestamp'] >= train_time_max].reset_index(drop=True)
    val_user_set = np.random.choice(np.arange(user_num), int(user_num / 2), replace=False)
    # test_user_set = np.
    # val_data = rest_data[rest_data['user'] < user_num / 2].reset_index(drop=True)
    val_data = rest_data[rest_data['user'].isin(val_user_set)].reset_index(drop=True)
    test_data = rest_data[~rest_data['user'].isin(val_user_set)].reset_index(drop=True)
    # print(train_data,val_data,test_data)

    # 划分高评分测试集
    test_data_high_rating = test_data.loc[test_data['rating'] > 4].reset_index(drop=True)
    val_data_high_rating = val_data.loc[val_data['rating'] > 4].reset_index(drop=True)

    print("saving train_data.csv, val_data.csv, test_data.csv")
    all_data.to_csv('data/' + dataset + '/all_data.csv', index=0, sep='\t')
    train_data.to_csv('data/' + dataset + '/train_data.csv', index=0, sep='\t')
    val_data.to_csv('data/' + dataset + '/val_data.csv', index=0, sep='\t')
    test_data.to_csv('data/' + dataset + '/test_data.csv', index=0, sep='\t')

    print("saving train_list.txt, val_list.txt, test_list.txt")
    # train_list.txt
    train_data = train_data.sort_values(by=['user']).reset_index(drop=True)
    train_user_count = train_data.groupby('user')['user'].count()
    f_train = open('data/' + dataset + '/train_list.txt', 'w')
    u = train_data['user'][0]
    user_interaction_num = train_data.groupby('user')['user'].count()[u]
    f_train.write(str(u) + '\t' + str(user_interaction_num) + '\t' + str(train_data['item'][0]))
    for i in range(1, train_data.shape[0]):
        if train_data['user'][i] == train_data['user'][i - 1]:
            f_train.write('\t' + str(train_data['item'][i]))
        else:
            u = train_data['user'][i]
            user_interaction_num = train_user_count[u]
            f_train.write('\n' + str(u) + '\t' + str(user_interaction_num) + '\t' + str(train_data['item'][i]))
    f_train.close()

    def test_process(test_data, test_data_high_rating, val=True):
        # test_list.txt
        test_data = test_data.sort_values(by=['user']).reset_index(drop=True)
        # print("test_data:\n", test_data)
        if val:
            f_test = open('data/' + dataset + '/val_list.txt', 'w')
        else:
            f_test = open('data/' + dataset + '/test_list.txt', 'w')
        f_test.write(str(test_data['user'][0]) + '\t' + str(test_data['item'][0]))
        for i in range(1, test_data.shape[0]):
            if test_data['user'][i] == test_data['user'][i - 1]:
                f_test.write('\t' + str(test_data['item'][i]))
            else:
                f_test.write('\n' + str(test_data['user'][i]) + '\t' + str(test_data['item'][i]))
        f_test.close()

        # test_high_rating_list.txt
        test_data_high_rating = test_data_high_rating.sort_values(by=['user']).reset_index(drop=True)
        # print("test_data_high_rating:\n", test_data_high_rating)
        if val:
            f_test = open('data/' + dataset + '/val_high_rating_list.txt', 'w')
        else:
            f_test = open('data/' + dataset + '/test_high_rating_list.txt', 'w')
        f_test.write(str(test_data_high_rating['user'][0]) + '\t' + str(test_data_high_rating['item'][0]))
        for i in range(1, test_data_high_rating.shape[0]):
            if test_data_high_rating['user'][i] == test_data_high_rating['user'][i - 1]:
                f_test.write('\t' + str(test_data_high_rating['item'][i]))
            else:
                f_test.write(
                    '\n' + str(test_data_high_rating['user'][i]) + '\t' + str(test_data_high_rating['item'][i]))
        f_test.close()

        # test_list_rating.txt
        if val:
            f_test_rating = open('data/' + dataset + '/val_list_rating.txt', 'w')
        else:
            f_test_rating = open('data/' + dataset + '/test_list_rating.txt', 'w')
        f_test_rating.write(str(test_data['user'][0]) + '\t' + str(test_data['rating'][0]))
        for i in range(1, test_data.shape[0]):
            if test_data['user'][i] == test_data['user'][i - 1]:
                f_test_rating.write('\t' + str(test_data['rating'][i]))
            else:
                f_test_rating.write('\n' + str(test_data['user'][i]) + '\t' + str(test_data['rating'][i]))
        f_test_rating.close()

    test_process(val_data, val_data_high_rating, val=True)
    test_process(test_data, test_data_high_rating, val=False)

    # 统计并记录每个item的交互数量和历史交互时间
    print("saving item_interactions.csv")
    train_data = train_data.sort_values(by=['item', 'timestamp']).reset_index(drop=True)
    f_item_interaction = open('data/' + dataset + '/item_interactions.csv', 'w')
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
    str_to_write = str(item_id) + ',' + str(interaction_num) + ',' + str_to_write + '\n'  # 最后一个item
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


