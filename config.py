import time
import os
import utils

class DefaultConfig(object):
    def __init__(self):
        # 基本配置
        self.backbone = 'MF'  # 'LightGCN'
        self.dataset = 'Ciao'  # 'Douban-movie' 'Amazon-CDs_and_Vinyl' 'Ciao' 'Amazon-Music' 'Amazon-Health'
        # self.dataset = 'Douban-movie'  # 'Douban-movie' 'Amazon-CDs_and_Vinyl' 'Ciao' 'Amazon-Music' 'Amazon-Health'
        self.method = 'TIDE'  # 'base' 'IPS' 'DICE' 'PD' 'PDA' 'TIDE' 'TIDE-fixq'
        self.val = True
        self.test_only = False
        self.show_performance = True
        self.data_process = False

        # 模型加载路径
        self.model_path_to_load = None

        # 数据集基础参数
        self.dataset_params = {
            'Douban-movie': {'lr': 0.001, 'n_u_f': 10, 'n_i_f': 10},
            'Amazon-CDs_and_Vinyl': {'lr': 0.001, 'n_u_f': 5, 'n_i_f': 5},
            'Amazon-Health': {'lr': 0.001, 'n_u_f': 5, 'n_i_f': 5},
            'Ciao': {'lr': 0.001, 'n_u_f': 5, 'n_i_f': 5},
            # 'Amazon-Music': {'lr': 0.001, 'lamb': 0.00, 'n_u_f': 5, 'n_i_f': 5},
            # 'gowalla': {'lr': 0.0001, 'lamb': 0.0001, 'n_u_f': 5, 'n_i_f': 5},
        }

        # 方法专用参数
        self.method_params = {
            'IPS': {'IPS_lambda': {
                'MF': {'Douban-movie': 10, 'Amazon-CDs_and_Vinyl': 30, 'Amazon-Music': 40,
                    'Ciao': 30, 'gowalla': 30, 'Amazon-Health': 30},
                'LightGCN': {'Douban-movie': 30, 'Amazon-CDs_and_Vinyl': 30, 'Amazon-Music': 30,
                            'Ciao': 30, 'gowalla': 30, 'Amazon-Health': 30}
            }},
            'PD': {'PDA_gamma': {'Douban-movie': 0.02, 'Ciao': 0.02,
                                'Amazon-CDs_and_Vinyl': 0.02, 'Amazon-Music': 0.06,
                                'Amazon-Health': 0.12}},  # 更新为最优 gamma
            'PDA': {'PDA_gamma': {'Douban-movie': 0.16, 'Ciao': 0.14,
                                'Amazon-CDs_and_Vinyl': 0.16, 'Amazon-Music': 0.04,
                                'Amazon-Health': 0.64},  # 更新为最优 gamma
                    'PDA_alpha': {'Douban-movie': 0.01, 'Ciao': 0.2,
                                'Amazon-CDs_and_Vinyl': 0.05, 'Amazon-Music': 0.2,
                                'Amazon-Health': 0.1}},  # alpha 保持原值
            'TIDE': {
                'tau': 1e7,
                'params': {
                    'MF': {
                        'Amazon-CDs_and_Vinyl': {'q': -2, 'b': -3, 'lr_q': 0.001, 'lr_b': 0.0001, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Amazon-Music': {'q': -1, 'b': -2, 'lr_q': 0.001, 'lr_b': 0.0001, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Ciao': {'q': -1, 'b': -1, 'lr_q': 0.1, 'lr_b': 0.001, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Douban-movie': {'q': -1, 'b': -5, 'lr_q': 0.1, 'lr_b': 0.0001, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'gowalla': {'q': -2, 'b': -5, 'lr_q': 0.001, 'lr_b': 1e-5, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Amazon-Health': {'q': -1, 'b': -2, 'lr_q': 0.001, 'lr_b': 0.001, 'lamb_q': 0.0, 'lamb_b': 0.0}  # 更新
                    },
                    'LightGCN': {
                        'Amazon-CDs_and_Vinyl': {'q': -1, 'b': -4, 'lr_q': 0.0001, 'lr_b': 0.001, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Amazon-Music': {'q': 0, 'b': -3, 'lr_q': 0.01, 'lr_b': 0.01, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Ciao': {'q': -1, 'b': -3, 'lr_q': 0.0001, 'lr_b': 0.0001, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Douban-movie': {'q': -2, 'b': -5, 'lr_q': 0.0001, 'lr_b': 1e-5, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'gowalla': {'q': -1, 'b': -3, 'lr_q': 0.01, 'lr_b': 0.0001, 'lamb_q': 0.0, 'lamb_b': 0.0},
                        'Amazon-Health': {'q': -1, 'b': -2, 'lr_q': 0.001, 'lr_b': 0.001, 'lamb_q': 0.03, 'lamb_b': 0.03}  # 更新
                    }
                }
            }
        }

        # 生成目录
        self.make_dir(f'log/{self.dataset}')
        self.get_file_path()

        # 初始化参数
        self.get_para()
        self.print_head_line()
        self.print_config_info()
        self.get_input_path()

    def get_basic_para(self):
        params = self.dataset_params.get(self.dataset, {})
        self.lr = params.get('lr', 0.001)
        self.lamb = params.get('lamb', 0.00)
        self.n_u_f = params.get('n_u_f', 5)
        self.n_i_f = params.get('n_i_f', 5)
        self.topk = 20
        self.epochs = 200
        self.batch_size = 8192 if self.backbone == 'MF' else 8192*4
        self.n_layers = 3 if self.backbone != 'MF' else None

    def get_ips_para(self):
        if self.method == 'IPS':
            self.IPS_lambda = self.method_params['IPS']['IPS_lambda'][self.backbone].get(self.dataset, 30)

    def get_pda_para(self):
        if self.method in ['PD', 'PDA']:
            if 'PDA_gamma' in self.method_params[self.method]:
                self.PDA_gamma = self.method_params[self.method]['PDA_gamma'].get(self.dataset, 0.02)
            if 'PDA_alpha' in self.method_params[self.method]:
                self.PDA_alpha = self.method_params[self.method]['PDA_alpha'].get(self.dataset, 0.1)

    def get_tide_para(self):
        if self.method.startswith('TIDE'):
            self.tau = self.method_params['TIDE']['tau']
            tide_dataset_params = self.method_params['TIDE']['params'][self.backbone].get(self.dataset, {})
            self.q = tide_dataset_params.get('q', -1)
            self.b = tide_dataset_params.get('b', -1)
            self.lr_q = tide_dataset_params.get('lr_q', 0.001)
            self.lr_b = tide_dataset_params.get('lr_b', 0.001)
            if self.method == 'TIDE-fixq':
                self.lr_q = 0

    def get_para(self):
        self.get_basic_para()
        self.get_ips_para()
        self.get_pda_para()
        self.get_tide_para()

    def get_file_path(self):
        str_time = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())
        log_dir = f'log/{self.dataset}'
        model_dir = f'log/{self.dataset}'
        self.make_dir(log_dir)
        self.make_dir(model_dir)

        self.log_path = f'{log_dir}/{str_time}_{self.backbone}_{self.method}.txt'
        self.model_path = f'{model_dir}/{str_time}_{self.backbone}_{self.method}.pth'
        self.q_path = f'{log_dir}/perf/q-{self.dataset}'
        self.make_dir(f'{log_dir}/perf')

        lgcn_graph_dir = f'./lgcn_graph/{self.dataset}'
        self.make_dir(lgcn_graph_dir)
        self.graph_index_path = f'{lgcn_graph_dir}/lgcn_graph_index.npy'
        self.graph_data_path = f'{lgcn_graph_dir}/lgcn_graph_data.npy'

        return self.log_path, self.model_path, self.q_path, self.graph_index_path, self.graph_data_path

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def print_head_line(self):
        method_str = f'backbone = {self.backbone}, method = {self.method}, dataset = {self.dataset}, val = {self.val}'
        self.print_str(method_str)

    def print_config_info(self):
        basic_info = f'lr = {self.lr:.4f}, lamb = {self.lamb}, batch_size = {self.batch_size}, topk = {self.topk}'
        self.print_str(basic_info)
        if self.backbone != 'MF':
            self.print_str(f'n_layers = {self.n_layers}')
        if self.method == 'IPS':
            self.print_str(f'IPS_lambda = {self.IPS_lambda}')
        if self.method in ['PD', 'PDA']:
            self.print_str(f'PDA_gamma = {self.PDA_gamma}')
            if self.method == 'PDA':
                self.print_str(f'PDA_alpha = {self.PDA_alpha}')
        if self.method.startswith('TIDE'):
            self.print_str(f'tau = {self.tau}, q = {self.q}, b = {self.b}, lr_q = {self.lr_q}, lr_b = {self.lr_b}')

    def print_str(self, str_to_print, file=True, window=True):
        utils.print_str(self.log_path, str_to_print, file, window)

    def get_input_path(self):
        main_path = 'data/'
        self.train_data = f'{main_path}{self.dataset}/train_data.csv'
        self.train_list = f'{main_path}{self.dataset}/train_list.txt'
        self.val_data = f'{main_path}{self.dataset}/val_data.csv'
        self.test_data = f'{main_path}{self.dataset}/test_data.csv'
        self.PDA_popularity_path = f'{main_path}{self.dataset}/PDA_popularity.csv'
        self.DICE_popularity_path = f'{main_path}{self.dataset}/DICE_popularity.npy'

        if self.val:
            self.test_list = f'{main_path}{self.dataset}/val_list.txt'
            self.test_list_h = f'{main_path}{self.dataset}/val_high_rating_list.txt'
            self.test_list_rating = f'{main_path}{self.dataset}/val_list_rating.txt'
        else:
            self.test_list = f'{main_path}{self.dataset}/test_list.txt'
            self.test_list_h = f'{main_path}{self.dataset}/test_high_rating_list.txt'
            self.test_list_rating = f'{main_path}{self.dataset}/test_list_rating.txt'


opt = DefaultConfig()
