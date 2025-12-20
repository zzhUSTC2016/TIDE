import cppimport
import numpy as np
import torch
from config import opt
import torch.nn.functional as F
import sys
import time
import pandas as pd

import utils
sys.path.append('cppcode')


class Evaluator(object):
    def __init__(self, model, user_num, item_num, max_time):
        self.model = model
        self.user_num = user_num
        self.item_num = item_num
        self.max_time = max_time
        self.pbd_import()
        # 分别记录在两个数据集上最好的测试结果
        self.best_perf = {'recall': np.zeros((1,)), 'precision': np.zeros((1,)),
                          'ndcg': np.zeros((1,)), 'averageRating': np.zeros((1,)),
                          'recall@3': np.zeros((1,)), 'precision@3': np.zeros((1,)),
                          'best_epoch': np.zeros((1,))}
        # 加载训练集和测试集中每个用户的交互记录， test阶段使用
        self.train_items = {}
        with open(opt.train_list) as f_train:
            for l in f_train.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split('\t')]
                uid, train_items_tmp = items[0], items[2:]
                self.train_items[uid] = train_items_tmp
        self.test_set = {}
        with open(opt.test_list) as f_test:
            for l in f_test.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split('\t')]
                uid, test_items = items[0], items[1:]
                self.test_set[uid] = test_items
        self.test_set_h = {}
        with open(opt.test_list_h) as f_test_h:
            for l in f_test_h.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [int(i) for i in l.split('\t')]
                uid, test_items = items[0], items[1:]
                self.test_set_h[uid] = test_items
        self.test_set_rating = {}
        with open(opt.test_list_rating) as f_test:
            for l in f_test.readlines():
                if len(l) == 0:
                    break
                l = l.strip('\n')
                items = [float(i) for i in l.split('\t')]
                uid, test_ratings = items[0], items[1:]
                self.test_set_rating[uid] = test_ratings

    def pbd_import(self):
        if opt.dataset == 'Douban-movie':
            self.pbd = cppimport.imp("pybind_douban")
        elif opt.dataset == 'Amazon-CDs_and_Vinyl':
            self.pbd = cppimport.imp("pybind_amazon_cd")
        elif opt.dataset == 'Amazon-Music':
            self.pbd = cppimport.imp("pybind_amazon_music")
        elif opt.dataset == 'Ciao':
            self.pbd = cppimport.imp("pybind_ciao")
        elif opt.dataset == 'Amazon-Health':
            self.pbd = cppimport.imp("pybind_amazon_health")
        else:
            raise ValueError("Dataset not supported.")

    def elu(self, x):
        return np.maximum(0, x) + np.minimum(0, np.exp(x) - 1) + 1

    def get_dcg(self, y_pred, y_true, k):
        dcg = np.zeros(len(y_true))
        for udx in range(len(y_true)):
            for i in range(1, k + 1):
                #  print(udx,i)
                if y_pred[udx][i - 1] in y_true[udx]:
                    dcg[udx] += 1 / np.log2(i + 1)
        return dcg

    def get_idcg(self, k):
        idcg = 0
        for i in range(1, k + 1):
            idcg += 1 / np.log2(i + 1)
        return idcg

    def get_ndcg(self, sorted_pred, test_items, k):
        dcg = self.get_dcg(sorted_pred, test_items, k)
        idcg = self.get_idcg(k)
        ndcg = dcg / idcg
        # print(np.mean(ndcg))
        return ndcg

    def auc(self, predicted_list, test_item_h):
        label = np.isin(predicted_list, test_item_h)
        m = np.sum(label == 1)
        n = np.sum(label == 0)
        if m == 0:
            return np.nan  # 0
        if n == 0:
            return np.nan  # 1
        # print(label)
        # print(np.arange(len(predicted_list), 0, -1))
        # print(np.arange(len(predicted_list), 0, -1) * label)
        auc = (np.sum(np.arange(len(predicted_list), 0, -1)
               * label) - (m * (m + 1) / 2)) / m / n
        # print(auc)
        return auc

    def evaluate(
            self,
            rate_batch,
            test_items,
            test_items_h,
            top_k,
            user_batch):
        # 存储每个user的recall, precision, ndcg。在两种测试集上的结果由最后一维区分。
        result = np.zeros((len(rate_batch), 6))
        pred = np.argpartition(
            rate_batch, -top_k)[:, -top_k:]   # 每个user的推荐列表，未排序
        sorted_pred = np.zeros(np.shape(pred))
        for udx in range(len(rate_batch)):
            '''print(rate_batch[udx,pred[udx]])
            print(np.argsort(rate_batch[udx,pred[udx]]))'''
            sorted_pred[udx] = pred[udx, np.argsort(
                rate_batch[udx, pred[udx]])][::-1]   # 排序后的推荐列表

        for udx in range(len(rate_batch)):
            u = user_batch[udx]
            hit = 0
            rating_sum = 0
            rating_vacant = 0
            for i in pred[udx]:
                if i in test_items[udx]:
                    hit += 1
                    idx = test_items[udx].index(i)
                    if self.test_set_rating[u][idx] == -1:
                        rating_vacant += 1
                    else:
                        rating_sum += self.test_set_rating[u][idx]

            # 得到recall、precision、average rating
            result[udx][0] = hit / len(test_items[udx])  # recall
            result[udx][1] = hit / top_k  # precision
            result[udx][3] = rating_sum / (hit - rating_vacant) if (
                hit - rating_vacant) > 0 else 0  # average_rating

        # 得到NDCG
        result[:, 2] = self.get_ndcg(sorted_pred, test_items, top_k)

        sorted_pred_2 = {}
        for udx in range(len(rate_batch)):
            # print(rate_batch[udx,pred[udx]])
            # print(np.argsort(rate_batch[udx,pred[udx]]))
            sorted_pred_2[udx] = np.array(test_items[udx])[
                np.argsort(-1 * rate_batch[udx, test_items[udx]])]  # 排序后的推荐列表

        top_k = 3
        for udx in range(len(rate_batch)):
            if len(test_items_h[udx]) < top_k or len(test_items[udx]) < top_k \
                    or len(test_items[udx]) - len(test_items_h[udx]) == 0:
                result[udx, 4] = np.nan
                result[udx, 5] = np.nan
            else:
                hit = 0
                for i in sorted_pred_2[udx][0:top_k]:
                    if i in test_items_h[udx]:
                        hit += 1
                # 得到recall@3、precision@3
                result[udx][4] = hit / len(test_items_h[udx])  # recall@3
                result[udx][5] = hit / top_k  # precision@3

        return result

    def test(self, users_to_test, popularity_item_np):
        USR_NUM, ITEM_NUM = self.user_num, self.item_num
        BATCH_SIZE = 8192 * 4
        top_show = np.array([opt.topk])
        max_top = max(top_show)
        result = {'precision': np.zeros(len(top_show)),
                  'recall': np.zeros(len(top_show)),
                  'ndcg': np.zeros(len(top_show)),
                  'averageRating': np.zeros(len(top_show)),
                  'recall@3': np.zeros(len(top_show)),
                  'precision@3': np.zeros(len(top_show))}
        u_batch_size = BATCH_SIZE  # 每个batch中user的数量
        test_users = users_to_test  # 测试集中出现的user
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        all_result = []
        item_batch = range(ITEM_NUM)  # 所有item都参与排序
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            # beta = model.beta.detach().cpu().numpy()
            # print("beta=", beta.mean(), beta.max(), beta.min())
            user_batch = test_users[start: end]
            user_batch = torch.tensor(user_batch).cuda()
            item_batch = torch.tensor(item_batch).cuda()
            embed_user, embed_item = self.model.get_emb(is_training=False)
            # embed_user, embed_item = self.model.embed_user, self.model.embed_item
            # embed_user, embed_item = self.model.embed_user_0.weight, self.model.embed_item_0.weight
            emb_u = embed_user[user_batch].detach().cpu().numpy()
            emb_i = embed_item[item_batch].detach().cpu().numpy()
            rate_batch = np.mat(emb_u) * np.mat(emb_i.T)  # embedding乘积矩阵

            rate_batch = np.array(rate_batch)
            # print(np.min(rate_batch), np.max(rate_batch), np.nanvar(rate_batch), np.mean(rate_batch))
            if opt.method == 'PD':
                rate_batch = self.elu(rate_batch)
            elif opt.method == 'PDA':
                rate_batch = self.elu(rate_batch) * \
                    (popularity_item_np ** opt.PDA_gamma)
            elif opt.method[0:4] == 'TIDE':
                rate_batch = np.log(1 + np.exp(rate_batch))  # 激活函数
                rate_batch = rate_batch * np.tanh(popularity_item_np)
            elif opt.method == 'p':  # 只用popularity进行推荐
                popularity_item_np = np.load(opt.DICE_popularity_path)
                rate_batch = np.zeros(
                    rate_batch.shape) + np.tanh(popularity_item_np)

            rate_batch = np.array(rate_batch)
            test_items = []  # 测试集数据
            test_items_h = []  # 高评分测试集数据
            user_batch = user_batch.cpu().numpy()
            item_batch = item_batch.cpu().numpy()

            for user in user_batch:
                test_items.append(self.test_set[user])
                if user in self.test_set_h.keys():
                    test_items_h.append(self.test_set_h[user])
                else:
                    test_items_h.append([])

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking
            # list.
            for idx, user in enumerate(user_batch):
                if user in self.train_items.keys():
                    train_items_off = self.train_items[user]
                    rate_batch[idx][train_items_off] = -np.inf
            batch_result = self.evaluate(
                rate_batch, test_items, test_items_h, max_top, user_batch)

            # batch_result = eval_score_matrix_foldout(rate_batch, test_items,
            # max_top)  # (B,k*metric_num), max_top= 20
            all_result.append(batch_result)

        # 每个batch的结果连接起来，对user求平均
        all_result = np.concatenate(all_result, axis=0)
        ratings = all_result[:, 3]
        final_result = np.nanmean(all_result, axis=0)  # mean
        result['recall'] += final_result[0]
        result['precision'] += final_result[1]
        result['ndcg'] += final_result[2]
        result['averageRating'] += np.sum(ratings) / \
            np.sum(n != 0 for n in ratings)
        result['recall@3'] += final_result[4]
        result['precision@3'] += final_result[5]
        return result

    def run_test(self, epoch):
        # 运行test()，并处理输出结果
        users_to_test = list(self.test_set.keys())
        if opt.method == 'PDA':
            PDA_popularity = pd.read_csv(opt.PDA_popularity_path, sep='\t')
            popularity_item_np = PDA_popularity['8'] + \
                opt.PDA_alpha * (PDA_popularity['8'] - PDA_popularity['7'])
            popularity_item_np = np.maximum(popularity_item_np.values, 0)
        elif opt.method[0:4] == 'TIDE':
            popularity_item_np = self.pbd.popularity(
                np.arange(
                    self.item_num), np.ones(
                    self.item_num,) * self.max_time)
            q = F.softplus(self.model.q).cpu().detach().numpy()
            b = F.softplus(self.model.b).cpu().detach().numpy()

            if opt.method == 'TIDE-noc' or opt.method == 'TIDE-int':
                popularity_item_np = q
            elif opt.method == 'TIDE-noq':
                popularity_item_np = b * popularity_item_np
            elif opt.method == 'TIDE-e':
                popularity_item_np = np.ones(popularity_item_np.shape)
            else:                                                   # TIDE-full
                popularity_item_np = q + b * popularity_item_np

            if opt.method[0:4] == 'TIDE' and opt.show_performance:
                str_tide_para_q = 'min(q) = %f, mean(q) = %f, max(q) = %f, var(q) = %f' % (
                    np.min(q), np.mean(q), np.max(q), np.nanvar(q))
                str_tide_para_b = 'min(b) = %f, mean(b) = %f, max(b) = %f, var(b) = %f' % (
                    np.min(b), np.mean(b), np.max(b), np.nanvar(b))
                utils.print_str(opt.log_path, str_tide_para_q)
                utils.print_str(opt.log_path, str_tide_para_b)
        else:
            popularity_item_np = 0

        ret = self.test(users_to_test, popularity_item_np)
        perf_str1 = 'recall=[%.5f], precision=[%.5f], ndcg=[%.5f]' % \
            (ret['recall'][0], ret['precision'][0], ret['ndcg'][0]
             )

        perf_str2 = 'recall@3=[%.5f], precision@3=[%.5f]' % \
                    (ret['recall@3'][0], ret['precision@3'][0])

        if opt.show_performance:
            utils.print_str(opt.log_path, perf_str1 + perf_str2)
            # util.print_str(opt.log_path, perf_str2)

        save_model_flag = 0
        if ret['recall'] + ret['precision'] + ret['ndcg'] > self.best_perf['recall'] + \
                self.best_perf['precision'] + self.best_perf['ndcg']:
            self.record_best_performance(ret, epoch)
            save_model_flag = 1

        return save_model_flag

    def record_best_performance(self, ret, epoch):
        self.best_perf['recall'] = ret['recall']
        self.best_perf['precision'] = ret['precision']
        self.best_perf['ndcg'] = ret['ndcg']
        self.best_perf['best_epoch'] = epoch
        self.best_perf['averageRating'] = ret['averageRating']
        self.best_perf['recall@3'] = ret['recall@3']
        self.best_perf['precision@3'] = ret['precision@3']

