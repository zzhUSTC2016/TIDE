import time
import cppimport
import torch
import numpy as np
import torch.optim as optim
import pandas as pd


def setup_seed(seed: int):
    """
    Set random seed for reproducibility.

    Parameters:
    - seed (int): random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_optimizer(model, opt):
    """
    Create and return an Adam optimizer for the model based on the backbone type.

    Parameters:
    - model: the PyTorch model to optimize
    - opt: an options object containing hyperparameters:
        - backbone: 'MF' or 'LightGCN'
        - lr, lr_q, lr_b: learning rates for different parameters
        - lamb_q, lamb_b: weight decay for q and b parameters

    Returns:
    - optimizer: a torch.optim.Adam optimizer
    """
    # 兼容参数未定义的情况
    lr = getattr(opt, 'lr', 0.001)
    lr_q = getattr(opt, 'lr_q', 0.001)
    lr_b = getattr(opt, 'lr_b', 0.001)
    lamb = getattr(opt, 'lamb', 0.0)        # 用户/物品嵌入正则化
    lamb_q = getattr(opt, 'lamb_q', 0.0)    # q 正则化
    lamb_b = getattr(opt, 'lamb_b', 0.0)    # b 正则化

    if getattr(opt, 'backbone', 'MF') == 'MF':
        params = [
            {"params": model.embed_user.weight, "lr": lr},
            {"params": model.embed_item.weight, "lr": lr},
            {"params": getattr(model, 'q', torch.tensor([])), "lr": lr_q, 'weight_decay': lamb_q},
            {"params": getattr(model, 'b', torch.tensor([])), "lr": lr_b, 'weight_decay': lamb_b}
        ]
    elif opt.backbone == 'LightGCN':
        params = [
            {"params": model.embed_user_0.weight, "lr": lr, 'weight_decay': lamb},
            {"params": model.embed_item_0.weight, "lr": lr, 'weight_decay': lamb},
            {"params": getattr(model, 'q', torch.tensor([])), "lr": lr_q, 'weight_decay': lamb_q},
            {"params": getattr(model, 'b', torch.tensor([])), "lr": lr_b, 'weight_decay': lamb_b}
        ]
    else:
        raise ValueError("Unsupported backbone: {}".format(getattr(opt, 'backbone', None)))


    optimizer = optim.Adam(params)
    return optimizer


def load_datasets(train_path: str, val_path: str, test_path: str):
    """
    Load train, validation, and test datasets.

    Parameters:
    - train_path (str): Path to the training data file (tab-separated)
    - val_path (str): Path to the validation data file (tab-separated)
    - test_path (str): Path to the test data file (tab-separated)

    Returns:
    - train_data, val_data, test_data: pandas DataFrames
    """
    train_data = pd.read_csv(train_path, sep='\t')
    val_data = pd.read_csv(val_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    return train_data, val_data, test_data


def get_dataset_info(train_data: pd.DataFrame,
                     val_data: pd.DataFrame,
                     test_data: pd.DataFrame):
    """
    Compute dataset statistics: number of users, number of items, min/max timestamp.

    Parameters:
    - train_data, val_data, test_data: pandas DataFrames

    Returns:
    - user_num (int): total number of users (max user ID + 1)
    - item_num (int): total number of items (max item ID + 1)
    - min_time (int/float): minimum timestamp in the training set
    - max_time (int/float): maximum timestamp in the training set
    """
    user_num = max(
        train_data['user'].max(),
        val_data['user'].max(),
        test_data['user'].max()
    ) + 1

    item_num = max(
        train_data['item'].max(),
        val_data['item'].max(),
        test_data['item'].max()
    ) + 1

    min_time = train_data['timestamp'].min()
    max_time = train_data['timestamp'].max()

    return user_num, item_num, min_time, max_time


def print_str(file_path: str, str_to_print: str, file: bool = True, window: bool = True):
    """
    Print string to log file and/or console window.

    Parameters:
    - file_path (str): path to log file
    - str_to_print (str): string content
    - file (bool): whether to write to file
    - window (bool): whether to print to console
    """
    if file:
        with open(file_path, 'a+') as f_log:
            f_log.write(str_to_print + '\n')
    if window:
        print(str_to_print)


def print_perf(best_perf: dict, log_path: str):
    """
    Print and log the best evaluation performance.

    Parameters:
    - best_perf (dict): expected keys ['recall', 'precision', 'ndcg', 'averageRating',
                                       'recall@3', 'precision@3', 'best_epoch']
                        values are expected to be 1-element arrays
    - log_path (str): path to the log file
    """
    best_epoch = int(best_perf['best_epoch'])
    recall = best_perf['recall'][0]
    precision = best_perf['precision'][0]
    ndcg = best_perf['ndcg'][0]
    avg_rating = best_perf['averageRating'][0]
    recall_3 = best_perf['recall@3'][0]
    precision_3 = best_perf['precision@3'][0]

    perf_str = (
        "End. Best epoch {:03d}: recall = {:.5f}, precision = {:.5f}, "
        "NDCG = {:.5f}, averageRating = {:.5f}"
    ).format(best_epoch, recall, precision, ndcg, avg_rating)

    perf_str_3 = "recall@3 = {:.5f}, precision@3 = {:.5f}".format(recall_3, precision_3)

    with open(log_path, 'a+') as f_log:
        f_log.write(perf_str + '\n' + perf_str_3 + '\n')

    print(perf_str)
    print(perf_str_3)


def log_epoch_loss(opt, epoch: int, elapsed_time: float, **loss_dict):
    """
    Log and print epoch losses and elapsed time.

    Parameters:
    - opt: options object containing log_path
    - epoch (int): current epoch number
    - elapsed_time (float): time spent for this epoch (seconds)
    - **loss_dict: named loss tensors (e.g., loss, loss_sum, loss_click, ...)
                   only losses that are not None will be logged
    """
    loss_parts = []

    for loss_name, loss_value in loss_dict.items():
        if loss_value is None:
            continue
        if hasattr(loss_value, "detach"):
            loss_scalar = loss_value.detach().cpu().numpy()
            if hasattr(loss_scalar, "__len__"):
                loss_scalar = loss_scalar[0]
        else:
            loss_scalar = loss_value
        loss_parts.append(f"{loss_name}: {loss_scalar:.5f}")

    loss_str = " | ".join(loss_parts)
    time_str = "The time elapse of epoch {:03d} is: {}".format(
        epoch, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    )

    log_path = getattr(opt, 'log_path', 'train.log')
    with open(log_path, 'a+') as f_log:
        f_log.write(loss_str + '\n')
        f_log.write(time_str + '\n\n')

    print(loss_str)
    print(time_str)


def import_pybind_module(dataset: str):
    """
    Import the corresponding pybind11 module based on dataset name.

    Parameters:
    - dataset (str): Name of the dataset. Supported datasets:
      'Douban-movie', 'Amazon-CDs_and_Vinyl', 'Amazon-Music', 'Ciao', 'Amazon-Health'

    Returns:
    - module: The imported pybind11 module

    Raises:
    - ValueError: if the dataset name is not supported
    """
    dataset_map = {
        'Douban-movie': 'pybind_douban',
        'Amazon-CDs_and_Vinyl': 'pybind_amazon_cd',
        'Amazon-Music': 'pybind_amazon_music',
        'Ciao': 'pybind_ciao',
        'Amazon-Health': 'pybind_amazon_health',
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset}")

    module_name = dataset_map[dataset]
    return cppimport.imp(module_name)
