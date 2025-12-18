from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import sys
import os

import utils
import train_utils
import models
from evaluate import Evaluator
from config import opt

# data process
from data.dataset import Data  

sys.path.append('cppcode')


def main():
    # Set random seed
    utils.setup_seed(20)

    # Load Pybind module
    pbd = utils.import_pybind_module(opt.dataset)

    # Load datasets and info
    train_data, val_data, test_data = utils.load_datasets(opt.train_data, opt.val_data, opt.test_data)
    user_num, item_num, min_time, max_time = utils.get_dataset_info(train_data, val_data, test_data)

    # Model & Optimizer
    model = getattr(models, 'Models')(user_num, item_num, 64, min_time, max_time)
    model.cuda()
    optimizer = utils.get_optimizer(model, opt)

    # Evaluator
    evaluator = Evaluator(model, user_num, item_num, max_time)


    if opt.val:
        pbd.load_user_interation_val()
    else:
        pbd.load_user_interation_test()

    # Test only mode
    if opt.test_only:
        model.load_state_dict(torch.load(opt.model_path_to_load))
        if opt.method == 'TIDE' and opt.backbone == 'MF':
            torch.save(model.q, opt.q_path)
        evaluator.run_test(0)
        return evaluator.best_perf


    # Training loop
    train_dataset = Data(train_data, item_num, 4, True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    best_perf = train_utils.train_model(opt, train_loader, model, optimizer, evaluator)

    # Print final best performance
    utils.print_perf(best_perf, opt.log_path)
    return best_perf




def test_after_training():
    opt.model_path_to_load = opt.model_path
    opt.test_only = True
    opt.val = False
    opt.get_input_path()
    print('---------testing----------')
    main()
    if opt.method == 'TIDE':
        '''opt.method = 'TIDE-e'
        print('---------testing TIDE-e----------')
        main()'''
        opt.method = 'TIDE-int'
        print('---------testing TIDE-int----------')
        main()


def main_base():
    lr_list = [opt.lr]
    lamb_list = [opt.lamb]
    '''lr_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    lamb_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]'''

    # lr_list = [0.1, 1, 10]
    # lamb_list = [0.00001, 0.0001, 0.001]

    '''lr_list = [0.00003, 0.0001, 0.0003]
    lamb_list = [0.003, 0.01]'''

    best_para = {'lr': 0, 'lamb': 0}
    best_perf = {'recall': np.zeros((1,)),
                 'precision': np.zeros((1,)),
                 'ndcg': np.zeros((1,))
                 }

    perf_df = pd.DataFrame(np.zeros((len(lamb_list), len(lr_list))),
                           columns=[str(j) for j in lr_list],
                           index=[str(j) for j in lamb_list])

    if (len(lr_list) == 1) and (len(lamb_list) == 1):
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = False

        for opt.lamb in lamb_list:
            for opt.lr in lr_list:
                str_para = 'lr = %f, lamb = %f' % (opt.lr, opt.lamb)
                utils.print_str(opt.log_path, str_para)
                perf = main()
                perf_df[str(opt.lr)][str(opt.lamb)] = perf['ndcg']
                if perf['ndcg'] > best_perf['ndcg']:
                    best_perf['recall'], best_perf['precision'], best_perf['ndcg'] = \
                        perf['recall'], perf['precision'], perf['ndcg']
                    best_para['lr'], best_para['lamb'] = opt.lr, opt.lamb

            str_best_para = 'Best parameter: lr = %f, lamb = %f' % (best_para['lr'], best_para['lamb'])
            str_best_perf = 'Best performance: rec = %.5f, pre = %.5f, ndcg = %.5f' % (
                best_perf['recall'], best_perf['precision'], best_perf['ndcg'])
            utils.print_str(opt.log_path, str_best_para)
            utils.print_str(opt.log_path, str_best_perf)
            print(perf_df)


def main_IPS():
    # IPS_lambda_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    IPS_lambda_list = [opt.IPS_lambda]

    if len(IPS_lambda_list) == 1:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = False
        for opt.IPS_lambda in IPS_lambda_list:
            print('\nIPS_lambda = %d' % opt.IPS_lambda)
            with open(opt.log_path, 'a+') as f_log:
                f_log.write('\nIPS_lambda = %d\t' % opt.IPS_lambda)
            main()


def main_PD():
    '''lr_list = [opt.lr]
    lamb_list = [opt.lamb]'''
    '''lr_list = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    lamb_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]'''

    # PDA_gamma_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
    PDA_gamma_list = [opt.PDA_gamma]

    best_para = {'PDA_gamma': 0}
    best_perf = {'recall': np.zeros((1,)),
                 'precision': np.zeros((1,)),
                 'ndcg': np.zeros((1,))
                 }

    perf_pd = pd.DataFrame(np.zeros((1, len(PDA_gamma_list))), columns=[str(j) for j in PDA_gamma_list])

    if len(PDA_gamma_list) == 1:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = False
        for opt.PDA_gamma in PDA_gamma_list:
            str_para = 'lr = %f, lamb = %f, PDA_gamma = %.2f' % (
                  opt.lr, opt.lamb, opt.PDA_gamma)
            utils.print_str(opt.log_path, str_para)
            perf = main()
            perf_pd[str(opt.PDA_gamma)] = perf['ndcg']
            if perf['ndcg'] > best_perf['ndcg']:
                best_perf['recall'], best_perf['precision'], best_perf['ndcg'] = \
                    perf['recall'], perf['precision'], perf['ndcg']
                best_para['PDA_gamma'] = opt.PDA_gamma

        str_best_para = 'Best parameter: PDA_gamma = %f' % (best_para['PDA_gamma'])
        str_best_perf = 'Best performance: rec = %.5f, pre = %.5f, ndcg = %.5f' % (
            best_perf['recall'], best_perf['precision'], best_perf['ndcg'])
        utils.print_str(opt.log_path, str_best_para)
        utils.print_str(opt.log_path, str_best_perf)
        print(perf_pd)


def main_PDA():
    PDA_gamma_list = [opt.PDA_gamma]
    PDA_alpha_list = [opt.PDA_alpha]
    # PDA_gamma_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
    # PDA_gamma_list.reverse()
    # PDA_alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    # PDA_alpha_list = [0.35, 0.4]

    best_para = {'PDA_gamma': 0, 'PDA_alpha': 0}
    best_perf = {'recall': np.zeros((1,)),
                 'precision': np.zeros((1,)),
                 'ndcg': np.zeros((1,))
                 }

    perf_pd = pd.DataFrame(np.zeros((len(PDA_alpha_list), len(PDA_gamma_list))),
                           columns=[str(j) for j in PDA_gamma_list],
                           index=[str(j) for j in PDA_alpha_list])

    if (len(PDA_gamma_list) == 1) and (len(PDA_alpha_list) == 1):
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = False
        for opt.PDA_gamma in PDA_gamma_list:
            for opt.PDA_alpha in PDA_alpha_list:
                str_para = 'lr = %f, lamb = %f, PDA_gamma = %.2f, PDA_alpha = %.2f' % (
                      opt.lr, opt.lamb, opt.PDA_gamma, opt.PDA_alpha)
                utils.print_str(opt.log_path, str_para)
                perf = main()
                perf_pd[str(opt.PDA_gamma)][str(opt.PDA_alpha)] = perf['ndcg']
                if perf['ndcg'] > best_perf['ndcg']:
                    best_perf['recall'], best_perf['precision'], best_perf['ndcg'] = \
                        perf['recall'], perf['precision'], perf['ndcg']
                    best_para['PDA_gamma'], best_para['PDA_alpha'] = opt.PDA_gamma, opt.PDA_alpha

        str_best_para = 'Best parameter: PDA_gamma = %f, PDA_alpha = %f' % (best_para['PDA_gamma'], best_para['PDA_alpha'])
        str_best_perf = 'Best performance: rec = %.5f, pre = %.5f, ndcg = %.5f' % (
            best_perf['recall'], best_perf['precision'], best_perf['ndcg'])
        utils.print_str(opt.log_path, str_best_para)
        utils.print_str(opt.log_path, str_best_perf)
        print(perf_pd)


def main_TIDE():

    '''q_list = [-1, 0, 1]
    lr_q_list = [0.0001, 0.001, 0.01]
    b_list = [-3, -2, -1]
    lr_b_list = [0.0001, 0.001, 0.01]'''

    q_list = [opt.q]
    lr_q_list = [opt.lr_q]
    b_list = [opt.b]
    lr_b_list = [opt.lr_b]

    '''q_list = [-1]
    lr_q_list = [0.0001, 0.001, 0.01]
    b_list = [-3, -2, -1]
    lr_b_list = [0.0001, 0.001, 0.01]'''

    if len(q_list) == 1 and len(lr_q_list) == 1 and len(b_list) == 1 and len(lr_b_list) == 1:
        opt.show_performance = True
        main()
        if not opt.test_only:
            test_after_training()
    else:
        opt.show_performance = False
        for opt.q in q_list:
            for opt.b in b_list:
                for opt.lr_q in lr_q_list:
                    for opt.lr_b in lr_b_list:
                        print('\nlr = %f, lamb = %f, q = %.2f, b = %.2f, lr_q = %f, lr_b = %f' %
                              (opt.lr, opt.lamb, opt.q, opt.b, opt.lr_q, opt.lr_b))
                        with open(opt.log_path, 'a+') as f_log:
                            f_log.write(
                                '\nlr = %f, lamb = %f, q = %.2f, b = %.2f, lr_q = %f, lr_b = %f\t' %
                              (opt.lr, opt.lamb, opt.q, opt.b, opt.lr_q, opt.lr_b))
                        main()


# 运行某个方法，搜索最优参数（可选）
if opt.method == 'IPS':
    main_IPS()
elif opt.method == 'PD':
    main_PD()
elif opt.method == 'PDA':
    main_PDA()
elif opt.method == 'TIDE':
    # main_TIDE()
    main_TIDE()
else:
    main_base()


# 在所有数据集的测试集上运行所有方法
def test_all_log_file():
    opt.val = False
    opt.test_only = True
    datasets = ['Ciao']  # 'Amazon-CDs_and_Vinyl', 'Douban-movie', 'Amazon-Music',
    sort_index = ['MF-base', 'MF-IPS', 'MF-DICE', 'MF-PD', 'MF-PDA', 'MF-TIDE',
                  'MF-TIDE-e', 'MF-TIDE-int', 'MF-TIDE-noc', 'MF-TIDE-noq', 'MF-TIDE-fixq',
                  'LightGCN-base', 'LightGCN-PD', 'LightGCN-PDA', 'LightGCN-TIDE']
    metrics = ['Rec@%d' % opt.topk, 'Pre@%d' % opt.topk, 'Ndcg@%d' % opt.topk, 'Rec@3', 'Pre@3']
    for dataset in datasets:
        opt.dataset = dataset
        opt.get_file_path()
        Evaluator.pbd_import(Evaluator)
        path = "./log_" + dataset  # 文件夹目录
        files = os.listdir(path)  # 得到文件夹下的所有文件名
        perf_pd = pd.DataFrame(columns=metrics)
        for file_name in files:
            if file_name[-4:] == '.txt':
                file_name = file_name[:-4]
                str_time, opt.backbone, opt.method = file_name.split('_')
                print(dataset, opt.backbone, opt.method)
                model_dir = 'model/' + opt.dataset
                model_path = model_dir + '/' + str_time + '_' + \
                             opt.backbone + "_" + opt.method + '.pth'
                opt.model_path_to_load = model_path
                opt.get_input_path()
                opt.get_para()

                perf = main()
                perf.pop('averageRating')
                perf.pop('best_epoch')
                perf = list(perf.values())
                perf = [i[0] for i in perf]
                perf_pd.loc[opt.backbone + '-' + opt.method] = perf
                # print(perf_pd)
                if opt.backbone == 'MF' and opt.method == 'TIDE':
                    for opt.method in ['TIDE-int', 'TIDE-e']:
                        perf = main()
                        perf.pop('averageRating')
                        perf.pop('best_epoch')
                        perf = list(perf.values())
                        perf = [i[0] for i in perf]
                        perf_pd.loc[opt.backbone + '-' + opt.method] = perf

                # print(perf_pd)
        perf_pd = perf_pd.loc[sort_index]
        perf_pd = perf_pd.round(5)
        perf_pd.to_csv('./perf/' + dataset + '_perf.csv', sep='\t')
        print(dataset)
        print(perf_pd)


# 对不同的K运行测试
def test_all_log_file_through_k():
    topk_list = [5, 10, 20, 50, 100]
    opt.val = False
    opt.test_only = True
    metric = 'ndcg'
    datasets = ['Ciao']  # 'Amazon-CDs_and_Vinyl', 'Douban-movie', 'Amazon-Music',
    sort_index = ['MF-base', 'MF-IPS', 'MF-DICE', 'MF-PD', 'MF-PDA', 'MF-TIDE',
                  'LightGCN-base', 'LightGCN-PD', 'LightGCN-PDA', 'LightGCN-TIDE']
    metrics = ['%s@%d' % (metric, i) for i in topk_list]
    for dataset in datasets:
        opt.dataset = dataset
        opt.get_file_path()
        Evaluator.pbd_import(Evaluator)
        path = "./log_" + dataset  # 文件夹目录
        files = os.listdir(path)  # 得到文件夹下的所有文件名
        perf_pd = pd.DataFrame(columns=metrics)
        for file_name in files:
            if file_name[-4:] == '.txt':
                file_name = file_name[:-4]
                str_time, opt.backbone, opt.method = file_name.split('_')
                print(dataset, opt.backbone, opt.method)
                model_dir = 'model/' + opt.dataset
                model_path = model_dir + '/' + str_time + '_' + \
                             opt.backbone + "_" + opt.method + '.pth'
                opt.model_path_to_load = model_path
                opt.get_input_path()
                opt.get_para()
                perf_list = []
                for topk in topk_list:
                    opt.topk = topk
                    perf = main()
                    '''perf.pop('averageRating')
                    perf.pop('best_epoch')
                    perf = list(perf.values())
                    perf = [i[0] for i in perf]'''
                    perf_list.append(perf[metric][0])
                perf_pd.loc[opt.backbone + '-' + opt.method] = perf_list
                # print(perf_pd)
        perf_pd = perf_pd.loc[sort_index]
        perf_pd = perf_pd.round(5)
        perf_pd.to_csv('./perf/' + metric + '_' + dataset + '_perf-through-k.csv', sep='\t')
        print(dataset)
        print(perf_pd)


# 运行上面两个函数
# test_all_log_file()
# test_all_log_file_through_k()
