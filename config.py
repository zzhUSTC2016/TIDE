import time
import os

import util


class DefaultConfig(object):
    """
    定义所有参数，包括运行那个数据集，选择哪个方法，所需文件的路径，每个方法对应的参数
    """
    def __init__(self):
        self.backbone = 'LightGCN'  # 'LightGCN'
        # 'Amazon-CDs_and_Vinyl' 'Amazon-Music' 'Douban-movie', 'Ciao'
        self.dataset = 'Ciao'
        # ['base','IPS','DICE','PD','PDA','TIDE', TIDE-noc, TIDE-noq, 'TIDE-e','TIDE-int']
        self.method = 'TIDE'
        self.val = True
        self.test_only = False
        self.show_performance = True
        self.data_process = False


        self.model_path_to_load = 'model/' + self.dataset + '/' + \
            '2022-03-17 20.41.18_MF_TIDE' + '.pth'
        self.get_file_path()
        self.print_head_line()
        self.get_para()
        self.print_config_info()
        self.get_input_path()

    def get_file_path(self):
        """
        To create file path for log and model saving
        """
        str_time = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())

        log_dir = 'log/' + self.dataset
        model_dir = 'model/' + self.dataset
        self.make_dir(log_dir)
        self.make_dir(model_dir)

        log_path = log_dir + '/' + str_time + '_' + self.backbone + "_" + self.method
        if not self.val:
            log_path += '_test'
        log_path += '.txt'

        model_path = model_dir + '/' + str_time + '_' + \
            self.backbone + "_" + self.method + '.pth'

        final_log_dir = 'log_' + self.dataset
        self.make_dir(final_log_dir)
        q_path = 'perf' + '/q' + '-%s' % self.dataset

        lgcn_graph_dir = './lgcn_graph/' + self.dataset
        self.make_dir(lgcn_graph_dir)
        graph_index_path = lgcn_graph_dir + '/lgcn_graph_index.npy'
        graph_data_path = lgcn_graph_dir + '/lgcn_graph_data.npy'

        self.log_path, self.model_path, self.q_path, \
            self.graph_index_path, self.graph_data_path = \
            log_path, model_path, q_path, graph_index_path, graph_data_path
        return log_path, model_path, q_path, graph_index_path, graph_data_path

    def make_dir(self, path):
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.makedirs(path)

    def print_head_line(self):
        if self.test_only:
            model_path_str = 'model_path = %s' % self.model_path_to_load
            self.print_str(model_path_str, window=False)
        method_str = 'backbone = %s, method = %s, dataset = %s, val = %s' % (
            self.backbone, self.method, self.dataset, self.val)
        self.print_str(method_str)

    def print_config_info(self):
        basic_info = 'lr = %.4f, lamb = %f, batch_size = %d, topk = %d' % (
            self.lr, self.lamb, self.batch_size, self.topk)
        self.print_str(basic_info)

        if self.backbone == 'LightGCN':
            layer_info = 'n_layers = %d' % self.n_layers
            self.print_str(layer_info)

        if self.method == 'IPS':
            ips_info = 'IPS_lambda = %d' % self.IPS_lambda
            self.print_str(ips_info)

        elif self.method == 'PD':
            pd_info = 'PD_gamma = %.2f' % self.PDA_gamma
            self.print_str(pd_info)

        elif self.method == 'PDA':
            pda_info = 'PDA_gamma = %.2f, PDA_alpha = %.2f' % (
                self.PDA_gamma, self.PDA_alpha)
            self.print_str(pda_info)

        elif self.method[0:4] == 'TIDE':
            tide_info = 'tau = %d, q = %.2f, b = %.2f\n' \
                        ' lr_q = %f, lr_b = %f' % (
                            self.tau, self.q, self.b, self.lr_q, self.lr_b)
            self.print_str(tide_info)

    def print_str(self, str_to_print, file=True, window=True):
        """
        To print string to log file and the window
        """
        util.print_str(self.log_path, str_to_print, file, window)

    def get_data_process_para(self):
        if self.dataset == 'Douban-movie':
            self.n_u_f, self.n_i_f = 10, 10  # 数据处理 最小交互数量
        else:
            self.n_u_f, self.n_i_f = 5, 5

    def get_basic_para_mf(self):
        self.topk = 20
        self.epochs = 200
        self.batch_size = 8192

        '''self.lr = 0.001  # 学习率
        self.lamb = 0.001

        if self.dataset == 'Ciao':
            self.lamb = 0.01  # 正则化系数  0.01
        elif self.dataset == 'Amazon-Music':
            if self.method == 'IPS':
                self.lamb = 0.03
            elif self.method == 'DICE':
                self.lr = 0.01
                self.lamb = 0.03
            else:
                self.lamb = 0.003'''

        if self.dataset == 'Douban-movie':
            self.lr = 0.001  # 学习率
            self.lamb = 0.001
        elif self.dataset == 'Amazon-CDs_and_Vinyl':
            self.lr = 0.001  # 学习率
            self.lamb = 0.001
        elif self.dataset == 'Amazon-Music':
            self.lr = 0.001  # 0.01  # 学习率
            self.lamb = 0.01  # 0.1
        elif self.dataset == 'Ciao':
            self.lr = 0.001  # 学习率
            self.lamb = 0.001
        elif self.dataset == 'gowalla':
            self.lr = 0.0001  # 学习率
            self.lamb = 0.0001

    def get_basic_para_lgcn(self):
        self.topk = 20
        self.epochs = 200
        self.batch_size = 8192 * 4
        self.n_layers = 3

        if self.dataset == 'Douban-movie':
            self.lr = 0.001  # 学习率
            self.lamb = 0.0001
        elif self.dataset == 'Amazon-CDs_and_Vinyl':
            self.lr = 0.01  # 学习率
            self.lamb = 0.001
        elif self.dataset == 'Amazon-Music':
            self.n_layers = 3
            self.batch_size = 8192
            self.lr = 0.01  # 0.1  # 学习率
            self.lamb = 0.01  # 0.0001
        elif self.dataset == 'Ciao':
            self.n_layers = 2
            self.batch_size = 8192
            self.lr = 0.01  # 0.01  # 学习率
            self.lamb = 0.01  # 0.000000001  # 0.1
        elif self.dataset == 'gowalla':
            self.n_layers = 3
            self.batch_size = 8192
            self.lr = 0.001  # 0.01  # 学习率
            self.lamb = 0.00001  # 0.1

    def get_basic_para(self):
        if self.backbone == 'MF':
            self.get_basic_para_mf()
        else:
            self.get_basic_para_lgcn()

    def get_ips_para_mf(self):
        if self.dataset == 'Douban-movie':
            self.IPS_lambda = 10
        elif self.dataset == 'Amazon-CDs_and_Vinyl':
            self.IPS_lambda = 30
        elif self.dataset == 'Amazon-Music':
            self.IPS_lambda = 40
        elif self.dataset == 'Ciao':
            self.IPS_lambda = 30
        elif self.dataset == 'gowalla':
            self.IPS_lambda = 30

    def get_ips_para_lgcn(self):
        if self.dataset == 'Douban-movie':
            self.IPS_lambda = 30
        elif self.dataset == 'Amazon-CDs_and_Vinyl':
            self.IPS_lambda = 30
        elif self.dataset == 'Amazon-Music':
            self.IPS_lambda = 30
        elif self.dataset == 'Ciao':
            self.IPS_lambda = 30
        elif self.dataset == 'gowalla':
            self.IPS_lambda = 30

    def get_ips_para(self):
        if self.backbone == 'MF':
            self.get_ips_para_mf()
        else:
            self.get_ips_para_lgcn()

    def get_tide_para_mf(self):
        if self.dataset == 'Amazon-CDs_and_Vinyl':
            self.q = -2  # 0.5
            self.lr_q = 0.001

            self.b = -3  # 5  # 2   # 0
            self.lr_b = 0.0001

        elif self.dataset == 'Amazon-Music':
            self.q = -1  # 0.5
            self.lr_q = 0.001

            self.b = -2  # 5  # 2   # 0
            self.lr_b = 0.0001

        elif self.dataset == 'Ciao':
            self.q = -1  # -1
            self.lr_q = 0.1  # 0.001

            self.b = -1  # -1
            self.lr_b = 0.001

        elif self.dataset == 'Douban-movie':
            self.q = -1  # 0
            self.lr_q = 0.1 # 0.0001

            self.b = -5
            self.lr_b = 0.0001
        elif self.dataset == 'gowalla':
            self.q = -2  # 0
            self.lr_q = 0.001

            self.b = -5
            self.lr_b = 0.00001

        self.tau = 1 * pow(10, 7)
        self.lamb_q = 0
        self.lamb_b = 0

    def get_tide_para_lgcn(self):
        if self.dataset == 'Amazon-CDs_and_Vinyl':
            self.q = -1  # 0.5
            self.lr_q = 0.0001

            self.b = -4  # 5  # 2   # 0
            self.lr_b = 0.001
        elif self.dataset == 'Amazon-Music':
            self.q = 0  # 0
            self.lr_q = 0.01

            self.b = -3
            self.lr_b = 0.01
        elif self.dataset == 'Ciao':
            self.q = -1  # -1  # -1
            self.lr_q = 0.0001  # 0.001

            self.b = -3  # -3  # -2    # 3
            self.lr_b = 0.0001  # 0.0001
        elif self.dataset == 'Douban-movie':
            self.q = -2  # 0
            self.lr_q = 0.0001

            self.b = -5
            self.lr_b = 0.00001
        elif self.dataset == 'gowalla':
            self.q = -1  # 0
            self.lr_q = 0.01

            self.b = -3
            self.lr_b = 0.0001
        self.tau = 1 * pow(10, 7)
        self.lamb_q = 0
        self.lamb_b = 0

    def get_tide_para(self):
        if self.backbone == 'MF':
            self.get_tide_para_mf()
        else:
            self.get_tide_para_lgcn()
        if self.method == 'TIDE-fixq':
            self.lr_q = 0

    def get_pda_para_mf(self):
        if self.method == 'PD':
            if self.dataset == 'Douban-movie':
                self.PDA_gamma = 0.02
            elif self.dataset == 'Ciao':
                self.PDA_gamma = 0.02
            elif self.dataset == 'Amazon-CDs_and_Vinyl':
                self.PDA_gamma = 0.02
            elif self.dataset == 'Amazon-Music':
                self.PDA_gamma = 0.06  # 0.12

        elif self.method == 'PDA':
            if self.dataset == 'Douban-movie':
                self.PDA_gamma = 0.16  # 0.12
                self.PDA_alpha = 0.01
            elif self.dataset == 'Ciao':
                self.PDA_gamma = 0.14  # 0.08
                self.PDA_alpha = 0.2  # 0.1
            elif self.dataset == 'Amazon-CDs_and_Vinyl':
                self.PDA_gamma = 0.16
                self.PDA_alpha = 0.05
            elif self.dataset == 'Amazon-Music':
                self.PDA_gamma = 0.04
                self.PDA_alpha = 0.2

    def get_pda_para_lgcn(self):
        if self.method == 'PD':
            if self.dataset == 'Douban-movie':
                self.PDA_gamma = 0.02
            elif self.dataset == 'Ciao':
                self.PDA_gamma = 0.02
            elif self.dataset == 'Amazon-CDs_and_Vinyl':
                self.PDA_gamma = 0.04
            elif self.dataset == 'Amazon-Music':
                self.PDA_gamma = 0.08

        elif self.method == 'PDA':
            if self.dataset == 'Douban-movie':
                self.PDA_gamma = 0.14  # 0.12
                self.PDA_alpha = 0.35
            elif self.dataset == 'Ciao':
                self.PDA_gamma = 0.08
                self.PDA_alpha = 0.05
            elif self.dataset == 'Amazon-CDs_and_Vinyl':
                self.PDA_gamma = 0.12
                self.PDA_alpha = 0.1
            elif self.dataset == 'Amazon-Music':
                self.PDA_gamma = 0.08
                self.PDA_alpha = 0.3

    def get_pda_para(self):
        if self.backbone == 'MF':
            self.get_pda_para_mf()
        else:
            self.get_pda_para_lgcn()

    def get_para(self):
        self.get_data_process_para()
        self.get_basic_para()
        self.get_ips_para()
        self.get_pda_para()
        self.get_tide_para()

    def get_input_path(self):
        main_path = 'data/'
        self.train_data = main_path + '{}/train_data.csv'.format(self.dataset)
        self.train_list = main_path + '{}/train_list.txt'.format(self.dataset)
        self.val_data = main_path + '{}/val_data.csv'.format(self.dataset)
        self.test_data = main_path + '{}/test_data.csv'.format(self.dataset)
        self.PDA_popularity_path = main_path + \
            '{}/PDA_popularity.csv'.format(self.dataset)
        self.DICE_popularity_path = main_path + \
            '{}/DICE_popularity.npy'.format(self.dataset)
        if self.val:
            self.test_list = main_path + '{}/val_list.txt'.format(self.dataset)
            self.test_list_h = main_path + \
                '{}/val_high_rating_list.txt'.format(self.dataset)
            self.test_list_rating = main_path + \
                '{}/val_list_rating.txt'.format(self.dataset)
        else:  # test
            self.test_list = main_path + \
                '{}/test_list.txt'.format(self.dataset)
            self.test_list_h = main_path + \
                '{}/test_high_rating_list.txt'.format(self.dataset)
            self.test_list_rating = main_path + \
                '{}/test_list_rating.txt'.format(self.dataset)


opt = DefaultConfig()
