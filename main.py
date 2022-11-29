from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import torch
import torch.optim as optim
import sys
import cppimport
import util
import os

import models
from evaluate import Evaluator
from config import opt
from data.dataset import Data

# import os
sys.path.append('cppcode')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_initializer(model):
    if opt.backbone == 'MF':
        '''params = [
            {
                "params": model.embed_user.weight, "lr": opt.lr, 'weight_decay': opt.lamb}, {
                "params": model.embed_item.weight, "lr": opt.lr, 'weight_decay': opt.lamb}, {
                "params": model.q, "lr": opt.lr_q, 'weight_decay': opt.lamb_q}, {
                "params": model.b, "lr": opt.lr_b, 'weight_decay': opt.lamb_b}]'''
        params = [
            {
                "params": model.embed_user.weight, "lr": opt.lr}, {
                "params": model.embed_item.weight, "lr": opt.lr}, {
                "params": model.q, "lr": opt.lr_q, 'weight_decay': opt.lamb_q}, {
                "params": model.b, "lr": opt.lr_b, 'weight_decay': opt.lamb_b}]
    elif opt.backbone == 'LightGCN':
        '''params = [
            {
                "params": model.embed_user_0.weight, "lr": opt.lr, 'weight_decay': opt.lamb}, {
                "params": model.embed_item_0.weight, "lr": opt.lr, 'weight_decay': opt.lamb}, {
                "params": model.q, "lr": opt.lr_q, 'weight_decay': opt.lamb_q}, {
                "params": model.b, "lr": opt.lr_b, 'weight_decay': opt.lamb_b}]'''
        params = [
            {
                "params": model.embed_user_0.weight, "lr": opt.lr}, {
                "params": model.embed_item_0.weight, "lr": opt.lr}, {
                "params": model.q, "lr": opt.lr_q, 'weight_decay': opt.lamb_q}, {
                "params": model.b, "lr": opt.lr_b, 'weight_decay': opt.lamb_b}]
    else:
        print("backbone error")
        raise
    optimizer = optim.Adam(params)
    return optimizer


def main():
    if opt.dataset == 'Douban-movie':
        pbd = cppimport.imp("pybind_douban")
    elif opt.dataset == 'Amazon-CDs_and_Vinyl':
        pbd = cppimport.imp("pybind_amazon_cd")
    elif opt.dataset == 'Amazon-Music':
        pbd = cppimport.imp("pybind_amazon_music")
    elif opt.dataset == 'Ciao':
        pbd = cppimport.imp("pybind_ciao")
    elif opt.dataset == 'gowalla':
        pbd = cppimport.imp("pybind_gowalla")

    setup_seed(20)
    train_data = pd.read_csv(opt.train_data, sep='\t')
    val_data = pd.read_csv(opt.val_data, sep='\t')
    test_data = pd.read_csv(opt.test_data, sep='\t')
    user_num = max(
        train_data['user'].max(),
        test_data['user'].max(),
        val_data['user'].max()) + 1
    item_num = max(
        train_data['item'].max(),
        test_data['item'].max(),
        val_data['item'].max()) + 1
    # print(user_num, item_num, len(train_data) + len(val_data) + len(test_data))
    min_time = train_data['timestamp'].min()
    max_time = train_data['timestamp'].max()

    # model = getattr(models, (opt.backbone + '_Model'))
    model = getattr(models, 'Models')
    train_dataset = Data(train_data, item_num, 4, True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0)

    model = model(user_num, item_num, 64, min_time, max_time)
    model.cuda()
    optimizer = get_initializer(model)
    evaluator = Evaluator(model, user_num, item_num, max_time)

    if opt.val:
        pbd.load_user_interation_val()
    else:
        pbd.load_user_interation_test()

    # loss_str_flag = 0

    if opt.test_only:
        model.load_state_dict(torch.load(opt.model_path_to_load))
        if opt.method == 'TIDE' and opt.backbone == 'MF':
            torch.save(model.q, opt.q_path)
        evaluator.run_test(0)
        return evaluator.best_perf

    # evaluator.run_test(0, popularity_item_PDA)
    DICE_alpha = 0.1 / 0.9
    for epoch in range(opt.epochs):
        # os.mknod(opt.model_path + '/%s.pth' % epoch)
        DICE_alpha *= 0.9
        model.train()  # 在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out。
        start_time = time.time()
        train_loader.dataset.ng_sample()  # 训练阶段，这一步生成真正的训练样本
        loss_sum = torch.tensor([0]).cuda()
        batch_id = 0
        for user, item_i, item_j, timestamp, split_idx in train_loader:
            batch_id += 1
            # print(batch_id)
            # batch_t0 = time.time()
            user = user.cuda().long()
            item_i = item_i.cuda().long()
            item_j = item_j.cuda().long()
            timestamp = timestamp.cpu().numpy()
            split_idx = split_idx.cpu().numpy()
            # model.zero_grad()
            optimizer.zero_grad()
            if opt.method == 'IPS':
                loss = model(user, item_i, item_j, timestamp, split_idx)
            elif opt.method == 'DICE':
                loss_click, loss_interest, loss_popularity_1, loss_popularity_2, loss_discrepancy, reg_loss = \
                    model(user, item_i, item_j, timestamp, split_idx)
                loss = loss_click + DICE_alpha * \
                    (loss_interest + loss_popularity_1 + loss_popularity_2) + 0.01 * loss_discrepancy
                loss += opt.lamb * reg_loss
            else:
                prediction_i, prediction_j, reg_loss = model(
                    user, item_i, item_j, timestamp, split_idx)  # 调用forward()方法
                # BPRloss
                # loss = -(prediction_i - prediction_j).sigmoid().log().sum()
                loss = torch.mean(torch.nn.functional.softplus(prediction_j - prediction_i))
                # print(loss, reg_loss)
                loss += opt.lamb * reg_loss

            loss.backward()  # 在这里得到梯度
            optimizer.step()  # 根据上面得到的梯度进行梯度回传。
            loss_sum = loss_sum + loss
            #
        # 一个epoch训练结束，开始测试

        if epoch % 1 == 0:
            model.eval()  # 测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out。
            save_model_flag = evaluator.run_test(epoch)
            if save_model_flag:
                open(opt.model_path, 'w').close()
                torch.save(model.state_dict(), opt.model_path)
                # torch.save(model.state_dict(), 'model/' + opt.method + '/model.pth')
                if opt.show_performance:
                    print("saved model")
        elapsed_time = time.time() - start_time
        if opt.show_performance:
            with open(opt.log_path, 'a+') as f_log:
                if opt.method == 'DICE':
                    f_log.write('Loss:%.1f, %.1f, %.1f, %.1f, %.1f, %.1f'
                                % (loss.detach().cpu().numpy(),
                                   loss_click.detach().cpu().numpy(),
                                   loss_interest.detach().cpu().numpy(),
                                   loss_popularity_1.detach().cpu().numpy(),
                                   loss_popularity_2.detach().cpu().numpy(),
                                   loss_discrepancy.detach().cpu().numpy()))
                else:
                    f_log.write(
                        'Loss:%f\n' %
                        loss_sum.detach().cpu().numpy()[0])
                f_log.write(
                    "The time elapse of epoch {:03d}".format(epoch) +
                    " is: " +
                    time.strftime(
                        "%H: %M: %S",
                        time.gmtime(elapsed_time)) +
                    '\n')
            if opt.method == 'DICE':
                print('Loss:', loss.detach().cpu().numpy(),
                      loss_click.detach().cpu().numpy(),
                      loss_interest.detach().cpu().numpy(),
                      loss_popularity_1.detach().cpu().numpy(),
                      loss_popularity_2.detach().cpu().numpy(),
                      loss_discrepancy.detach().cpu().numpy())
            else:
                print('Loss:', loss_sum.detach().cpu().numpy()[0])
            print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
                  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        if epoch - evaluator.best_perf['best_epoch'] > 10:  # and epoch > 50:
            with open(opt.log_path, 'a+') as f_log:
                f_log.write("early stop at %d epoch" % epoch)
            print("early stop at %d epoch" % epoch)
            break

    with open(opt.log_path, 'a+') as f_log:
        f_log.write(
            "\nEnd. Best epoch {:03d}: recall = {:.5f}, precision = {:.5f}, NDCG = {:.5f}, "
            "sum = {:.5f}, averageRating = {:.5f}".format(
                evaluator.best_perf['best_epoch'],
                evaluator.best_perf['recall'][0],
                evaluator.best_perf['precision'][0],
                evaluator.best_perf['ndcg'][0],
                evaluator.best_perf['recall'][0] +
                evaluator.best_perf['precision'][0] +
                evaluator.best_perf['ndcg'][0],
                evaluator.best_perf['averageRating'][0]))
        '''f_log.write(" recall = {:.5f}, precision = {:.5f}, "
                    "NDCG = {:.5f}\n".format(evaluator.best_perf_h['recall'][0],
                                             evaluator.best_perf_h['precision'][0],
                                             evaluator.best_perf_h['ndcg'][0]))'''
        f_log.write(
            "recall@3 = {:.5f}, precision@3 = {:.5f},\n".format(
                evaluator.best_perf['recall@3'][0],
                evaluator.best_perf['precision@3'][0]))
    print(
        "End. Best epoch {:03d}: recall = {:.5f}, precision = {:.5f}, NDCG = {:.5f}, "
        "averageRating = {:.5f}".format(
            evaluator.best_perf['best_epoch'],
            evaluator.best_perf['recall'][0],
            evaluator.best_perf['precision'][0],
            evaluator.best_perf['ndcg'][0],
            evaluator.best_perf['averageRating'][0]))
    '''print(" recall = {:.5f}, precision = {:.5f}, "
          "NDCG = {:.5f}".format(evaluator.best_perf_h['recall'][0],
                                   evaluator.best_perf_h['precision'][0],
                                   evaluator.best_perf_h['ndcg'][0]))'''
    print(
        " recall@3 = {:.5f}, precision@3 = {:.5f},".format(
            evaluator.best_perf['recall@3'][0],
            evaluator.best_perf['precision@3'][0]))
    return evaluator.best_perf


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

    # lr_list = [0.1]
    # lamb_list = [0.00001, 0.0001, 0.001]
    # lamb_list = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]

    # lr_list = [0.1, 1, 10]
    # lamb_list = [0.00001, 0.0001, 0.001]

    # lamb_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

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

    '''lr_list = [0.000001]
    lamb_list = [0.001]'''

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
                util.print_str(opt.log_path, str_para)
                perf = main()
                perf_df[str(opt.lr)][str(opt.lamb)] = perf['ndcg']
                if perf['ndcg'] > best_perf['ndcg']:
                    best_perf['recall'], best_perf['precision'], best_perf['ndcg'] = \
                        perf['recall'], perf['precision'], perf['ndcg']
                    best_para['lr'], best_para['lamb'] = opt.lr, opt.lamb

            str_best_para = 'Best parameter: lr = %f, lamb = %f' % (best_para['lr'], best_para['lamb'])
            str_best_perf = 'Best performance: rec = %.5f, pre = %.5f, ndcg = %.5f' % (
                best_perf['recall'], best_perf['precision'], best_perf['ndcg'])
            util.print_str(opt.log_path, str_best_para)
            util.print_str(opt.log_path, str_best_perf)
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
            util.print_str(opt.log_path, str_para)
            perf = main()
            perf_pd[str(opt.PDA_gamma)] = perf['ndcg']
            if perf['ndcg'] > best_perf['ndcg']:
                best_perf['recall'], best_perf['precision'], best_perf['ndcg'] = \
                    perf['recall'], perf['precision'], perf['ndcg']
                best_para['PDA_gamma'] = opt.PDA_gamma

        str_best_para = 'Best parameter: PDA_gamma = %f' % (best_para['PDA_gamma'])
        str_best_perf = 'Best performance: rec = %.5f, pre = %.5f, ndcg = %.5f' % (
            best_perf['recall'], best_perf['precision'], best_perf['ndcg'])
        util.print_str(opt.log_path, str_best_para)
        util.print_str(opt.log_path, str_best_perf)
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
                util.print_str(opt.log_path, str_para)
                perf = main()
                perf_pd[str(opt.PDA_gamma)][str(opt.PDA_alpha)] = perf['ndcg']
                if perf['ndcg'] > best_perf['ndcg']:
                    best_perf['recall'], best_perf['precision'], best_perf['ndcg'] = \
                        perf['recall'], perf['precision'], perf['ndcg']
                    best_para['PDA_gamma'], best_para['PDA_alpha'] = opt.PDA_gamma, opt.PDA_alpha

        str_best_para = 'Best parameter: PDA_gamma = %f, PDA_alpha = %f' % (best_para['PDA_gamma'], best_para['PDA_alpha'])
        str_best_perf = 'Best performance: rec = %.5f, pre = %.5f, ndcg = %.5f' % (
            best_perf['recall'], best_perf['precision'], best_perf['ndcg'])
        util.print_str(opt.log_path, str_best_para)
        util.print_str(opt.log_path, str_best_perf)
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
