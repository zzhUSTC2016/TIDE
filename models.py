import cppimport
import torch
import torch.nn as nn
from config import opt
import sys
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

sys.path.append('cppcode')


class Models(nn.Module):
    def __init__(self, user_num, item_num, factor_num, min_time, max_time):
        super(Models, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.n_users = user_num
        self.m_items = item_num
        self.embed_size = factor_num
        self.max_time = max_time
        self.min_time = min_time

        # 初始化模型参数
        self.q = nn.Parameter(torch.ones(item_num) * getattr(opt, 'q', 0.0))
        self.b = nn.Parameter(torch.ones(item_num) * getattr(opt, 'b', 0.0))
        self.tau = torch.ones(item_num) * getattr(opt, 'tau', 1e8)

        # 导入 C++ PBD 模块
        self.pbd_import()

        # TIDE 系列方法加载流行度
        if opt.method in ['TIDE', 'TIDE-noq', 'TIDE-fixq']:
            self.pbd.load_popularity(self.tau.cpu().detach().numpy())

        # 读取 PDA 与 DICE 流行度数据
        self.PDA_array = pd.read_csv(opt.PDA_popularity_path, sep='\t').values.reshape(-1)
        self.DICE_pop = np.load(opt.DICE_popularity_path)

        # Embedding 初始化
        if opt.backbone == 'LightGCN':
            self.get_graph()
            self.embed_user_0 = nn.Embedding(user_num, factor_num)
            self.embed_item_0 = nn.Embedding(item_num, factor_num)
            nn.init.normal_(self.embed_user_0.weight, std=0.1)
            nn.init.normal_(self.embed_item_0.weight, std=0.1)
            self.get_emb()
        else:
            self.embed_user = nn.Embedding(user_num, factor_num)
            self.embed_item = nn.Embedding(item_num, factor_num)
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.normal_(self.embed_item.weight, std=0.01)

    def pbd_import(self):
        """根据数据集选择 C++ PBD 模块"""
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
            raise ValueError(f"Dataset {opt.dataset} not supported")

    def get_graph(self):
        """生成 LightGCN 的图结构"""
        if os.path.exists(opt.graph_data_path) and os.path.exists(opt.graph_index_path):
            graph_data = torch.from_numpy(np.load(opt.graph_data_path))
            graph_index = torch.from_numpy(np.load(opt.graph_index_path))
            graph_size = torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])
            self.Graph = torch.sparse.FloatTensor(graph_index, graph_data, graph_size).coalesce().cuda()
        else:
            train_data = pd.read_csv(opt.train_data, sep='\t')
            trainUser = torch.LongTensor(train_data['user'].values)
            trainItem = torch.LongTensor(train_data['item'].values)

            first_sub = torch.stack([trainUser, trainItem + self.n_users])
            second_sub = torch.stack([trainItem + self.n_users, trainUser])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()

            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])).coalesce().cuda()

            np.save(opt.graph_index_path, index.t().numpy())
            np.save(opt.graph_data_path, data.numpy())

    def get_emb(self, is_training=True):
        """计算用户与物品的 embedding"""
        if opt.backbone == 'MF':
            return self.embed_user.weight, self.embed_item.weight
        else:
            users_emb = self.embed_user_0.weight
            items_emb = self.embed_item_0.weight
            all_emb = torch.cat([users_emb, items_emb]).cuda()
            embs = [all_emb]

            for layer in range(opt.n_layers):
                all_emb = torch.sparse.mm(self.Graph, all_emb)
                embs.append(all_emb)

            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            self.embed_user, self.embed_item = torch.split(light_out, [self.n_users, self.m_items])
            return self.embed_user, self.embed_item

    def reg_loss(self, user, item_i, item_j):
        """计算 L2 正则化"""
        if opt.backbone == 'LightGCN':
            user_embedding_0 = self.embed_user_0.weight[user]
            item_i_embedding_0 = self.embed_item_0.weight[item_i]
            item_j_embedding_0 = self.embed_item_0.weight[item_j]
        else:
            user_embedding_0 = self.embed_user.weight[user]
            item_i_embedding_0 = self.embed_item.weight[item_i]
            item_j_embedding_0 = self.embed_item.weight[item_j]

        reg_loss = (1 / 2) * (user_embedding_0.norm(2).pow(2) +
                              item_i_embedding_0.norm(2).pow(2) +
                              item_j_embedding_0.norm(2).pow(2)) / float(len(user))
        return reg_loss

    def forward(self, user, item_i, item_j, timestamp, split_idx):
        """模型前向计算，包括多种训练方法分支"""
        embed_user, embed_item = self.get_emb()
        user_embedding = embed_user[user]
        item_i_embedding = embed_item[item_i]
        item_j_embedding = embed_item[item_j]

        reg_loss = self.reg_loss(user, item_i, item_j)

        if opt.method == "base":  # MF模型，不用计算popularity
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            return prediction_i, prediction_j, reg_loss

        elif opt.method == 'IPS':
            item_i_np = item_i.cpu().numpy().astype(int)
            IPS_c = torch.from_numpy(np.minimum(1 / self.DICE_pop[item_i_np], opt.IPS_lambda)).cuda() / opt.IPS_lambda
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            ips_loss = -1 * (IPS_c * (prediction_i - prediction_j).sigmoid().log()).sum() + opt.lamb * reg_loss
            return ips_loss

        elif opt.method == 'DICE':
            DICE_size = int(self.embed_size / 2)
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            loss_click = -(prediction_i - prediction_j).sigmoid().log().sum()

            # 分拆 embedding 计算 discrepancy
            user_embedding_1 = self.embed_user(user.unique())[:, 0:DICE_size]
            item_i_embedding_1 = self.embed_item(item_i.unique())[:, 0:DICE_size]
            item_j_embedding_1 = self.embed_item(item_j.unique())[:, 0:DICE_size]
            user_embedding_2 = self.embed_user(user.unique())[:, DICE_size:]
            item_i_embedding_2 = self.embed_item(item_i.unique())[:, DICE_size:]
            item_j_embedding_2 = self.embed_item(item_j.unique())[:, DICE_size:]
            loss_discrepancy = -1 * ((user_embedding_1 - user_embedding_2).sum() +
                                     (item_i_embedding_1 - item_i_embedding_2).sum() +
                                     (item_j_embedding_1 - item_j_embedding_2).sum())

            # 按流行度划分正负样本
            item_i_np = item_i.cpu().numpy().astype(int)
            item_j_np = item_j.cpu().numpy().astype(int)
            pop_relation = self.DICE_pop[item_i_np] > self.DICE_pop[item_j_np]
            user_O1 = user[pop_relation]
            user_O2 = user[~pop_relation]
            item_i_O1 = item_i[pop_relation]
            item_j_O1 = item_j[pop_relation]
            item_i_O2 = item_i[~pop_relation]
            item_j_O2 = item_j[~pop_relation]

            # interest loss
            user_embedding = self.embed_user(user_O1)[:, 0:DICE_size]
            item_i_embedding = self.embed_item(item_i_O1)[:, 0:DICE_size]
            item_j_embedding = self.embed_item(item_j_O1)[:, 0:DICE_size]
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            loss_interest = -(prediction_i - prediction_j).sigmoid().log().sum()

            # popularity loss
            user_embedding = self.embed_user(user_O1)[:, DICE_size:]
            item_i_embedding = self.embed_item(item_i_O1)[:, DICE_size:]
            item_j_embedding = self.embed_item(item_j_O1)[:, DICE_size:]
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            loss_popularity_1 = -(prediction_j - prediction_i).sigmoid().log().sum()

            user_embedding = self.embed_user(user_O2)[:, DICE_size:]
            item_i_embedding = self.embed_item(item_i_O2)[:, DICE_size:]
            item_j_embedding = self.embed_item(item_j_O2)[:, DICE_size:]
            prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
            prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
            loss_popularity_2 = -(prediction_i - prediction_j).sigmoid().log().sum()

            return loss_click, loss_interest, loss_popularity_1, loss_popularity_2, loss_discrepancy, reg_loss

        elif opt.method in ["PDA", "PD"]:
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            PDA_idx_i = (item_i_np * 10 + split_idx).astype(int)
            PDA_idx_j = (item_j_np * 10 + split_idx).astype(int)
            PDA_popularity_i = torch.from_numpy(self.PDA_array[PDA_idx_i]).cuda()
            PDA_popularity_j = torch.from_numpy(self.PDA_array[PDA_idx_j]).cuda()
            prediction_i = (F.elu((user_embedding * item_i_embedding).sum(dim=-1)) + 1) * (PDA_popularity_i ** opt.PDA_gamma)
            prediction_j = (F.elu((user_embedding * item_j_embedding).sum(dim=-1)) + 1) * (PDA_popularity_j ** opt.PDA_gamma)
            return prediction_i, prediction_j, reg_loss

        elif opt.method == 'TIDE-noc':
            self.popularity_i = F.softplus(self.q[item_i])
            self.popularity_j = F.softplus(self.q[item_j])
            self.prediction_i = F.softplus((user_embedding * item_i_embedding).sum(dim=-1)) * torch.tanh(self.popularity_i)
            self.prediction_j = F.softplus((user_embedding * item_j_embedding).sum(dim=-1)) * torch.tanh(self.popularity_j)
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss

        elif opt.method == 'TIDE-noq':
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            popularity_i_s = torch.from_numpy(self.pbd.popularity(item_i_np, timestamp)).cuda()
            popularity_j_s = torch.from_numpy(self.pbd.popularity(item_j_np, timestamp)).cuda()
            self.popularity_i = F.softplus(self.b[item_i]) * popularity_i_s
            self.popularity_j = F.softplus(self.b[item_j]) * popularity_j_s
            self.prediction_i = F.softplus((user_embedding * item_i_embedding).sum(dim=-1)) * torch.tanh(self.popularity_i)
            self.prediction_j = F.softplus((user_embedding * item_j_embedding).sum(dim=-1)) * torch.tanh(self.popularity_j)
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss

        elif opt.method in ['TIDE', 'TIDE-int', 'TIDE-e', 'TIDE-fixq']:
            item_i_np = item_i.cpu().numpy()
            item_j_np = item_j.cpu().numpy()
            popularity_i_s = torch.from_numpy(self.pbd.popularity(item_i_np, timestamp)).cuda()
            popularity_j_s = torch.from_numpy(self.pbd.popularity(item_j_np, timestamp)).cuda()
            self.popularity_i = F.softplus(self.q[item_i]) + F.softplus(self.b[item_i]) * popularity_i_s
            self.popularity_j = F.softplus(self.q[item_j]) + F.softplus(self.b[item_j]) * popularity_j_s
            self.prediction_i = F.softplus((user_embedding * item_i_embedding).sum(dim=-1)) * torch.tanh(self.popularity_i)
            self.prediction_j = F.softplus((user_embedding * item_j_embedding).sum(dim=-1)) * torch.tanh(self.popularity_j)
            self.prediction_i.retain_grad()
            self.prediction_j.retain_grad()
            return self.prediction_i, self.prediction_j, reg_loss
