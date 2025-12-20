import cppimport
import numpy as np
import sys
from torch.utils.data import Dataset

from config import opt

sys.path.append('cppcode')
if opt.dataset == 'Douban-movie':
    pbd = cppimport.imp("pybind_douban")
elif opt.dataset == 'Amazon-CDs_and_Vinyl':
    pbd = cppimport.imp("pybind_amazon_cd")
elif opt.dataset == 'Amazon-Music':
    pbd = cppimport.imp("pybind_amazon_music")
elif opt.dataset == 'Ciao':
    pbd = cppimport.imp("pybind_ciao")
elif opt.dataset == 'Amazon-Health':
    pbd = cppimport.imp("pybind_amazon_health")
else:
    raise ValueError(f"Unsupported dataset: {opt.dataset}")


class Data(Dataset):
    def __init__(self, features,
                 num_item, num_ng=0, is_training=None):
        """features=train_data,num_item=item_num,train_mat，稀疏矩阵，num_ng,训练阶段默认为4，即采样4-1个负样本对应一个评分过的
        数据。
        """
        super(Data, self).__init__()
        """ Note that the labels are only useful when training, we thus
            add them in the ng_sample() function.
        """
        self.features = features  # 训练数据，pd.series类
        self.num_item = num_item    # 训练和测试数据中itemID的最大值
        self.num_ng = num_ng  # 训练阶段负采样的数量
        self.is_training = is_training

    def ng_sample(self):  # 用c语言模块做负采样
        assert self.is_training, 'no need to sampling when testing'
        self.features_fill = []  # 为每个正样本匹配4个负样本
        user_positive = np.array(self.features['user'])
        item_negative = pbd.negtive_sample(
            user_positive, np.array(
                self.num_ng))  # 用C语言为每个交互做k个负采样
        item_negative = item_negative.reshape(
            (item_negative.shape[0], 1))  # 变换为列向量

        user_positive = np.array(
            self.features['user']).repeat(
            self.num_ng)  # userID重复k次，为了与负样本列表大小匹配
        user_positive = user_positive.reshape((user_positive.shape[0], 1))
        item_positive = np.array(
            self.features['item']).repeat(
            self.num_ng)  # 正样本item做同样的操作
        item_positive = item_positive.reshape((item_positive.shape[0], 1))
        time_positive = np.array(
            self.features['timestamp']).repeat(
            self.num_ng)    # timestamp做同样的操作
        time_positive = time_positive.reshape((time_positive.shape[0], 1))
        split_idx = np.array(
            self.features['split_idx']).repeat(
            self.num_ng)  # timestamp做同样的操作
        split_idx = split_idx.reshape((split_idx.shape[0], 1))

        features_np = np.concatenate(
            (user_positive,
             item_positive,
             item_negative,
             time_positive,
             split_idx),
            axis=1)  # 将正样本和负样本合并
        self.features_fill = features_np.tolist()

    def __len__(self):
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        timestamp = features[idx][3]
        PDA_popularity = features[idx][4]
        return user, item_i, item_j, timestamp, PDA_popularity
