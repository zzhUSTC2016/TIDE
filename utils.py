import time
import cppimport
import torch
import numpy as np
import torch.optim as optim
import pandas as pd

def setup_seed(seed):
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
    
    if opt.backbone == 'MF':
        # Parameters for Matrix Factorization
        params = [
            {"params": model.embed_user.weight, "lr": opt.lr},
            {"params": model.embed_item.weight, "lr": opt.lr},
            {"params": model.q, "lr": opt.lr_q, 'weight_decay': opt.lamb_q},
            {"params": model.b, "lr": opt.lr_b, 'weight_decay': opt.lamb_b}
        ]
    elif opt.backbone == 'LightGCN':
        # Parameters for LightGCN
        params = [
            {"params": model.embed_user_0.weight, "lr": opt.lr},
            {"params": model.embed_item_0.weight, "lr": opt.lr},
            {"params": model.q, "lr": opt.lr_q, 'weight_decay': opt.lamb_q},
            {"params": model.b, "lr": opt.lr_b, 'weight_decay': opt.lamb_b}
        ]
    else:
        raise ValueError("Unsupported backbone: {}".format(opt.backbone))

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


def print_str(file_path, str_to_print, file=True, window=True):
    """
    To print string to log file and the window
    """
    if file:
        with open(file_path, 'a+') as f_log:
            f_log.write(str_to_print)
            f_log.write('\n')
    if window:
        print(str_to_print)


def print_perf(best_perf: dict, log_path: str):
    """
    Print and log the best evaluation performance.

    This function is designed to be fully compatible with the Evaluator.best_perf
    structure used in the project.

    Parameters:
    - best_perf (dict): best performance dictionary from Evaluator,
                        expected keys:
                        ['recall', 'precision', 'ndcg', 'averageRating',
                         'recall@3', 'precision@3', 'best_epoch']
    - log_path (str): path to the log file
    """

    # Extract values (Evaluator stores metrics as 1-element arrays)
    best_epoch = int(best_perf['best_epoch'])
    recall = best_perf['recall'][0]
    precision = best_perf['precision'][0]
    ndcg = best_perf['ndcg'][0]
    avg_rating = best_perf['averageRating'][0]
    recall_3 = best_perf['recall@3'][0]
    precision_3 = best_perf['precision@3'][0]

    # Main performance string (same as original code style)
    perf_str = (
        "\nEnd. Best epoch {:03d}: recall = {:.5f}, precision = {:.5f}, "
        "NDCG = {:.5f}, averageRating = {:.5f}"
    ).format(
        best_epoch,
        recall,
        precision,
        ndcg,
        avg_rating
    )

    # Additional @3 metrics
    perf_str_3 = (
        "recall@3 = {:.5f}, precision@3 = {:.5f}"
    ).format(
        recall_3,
        precision_3
    )

    # Write to log file
    with open(log_path, 'a+') as f_log:
        f_log.write(perf_str)
        f_log.write('\n' + perf_str_3 + '\n')

    # Print to console
    print(perf_str)
    print(perf_str_3)


def log_epoch_loss(opt, epoch, elapsed_time, **loss_dict):
    """
    Log and print epoch losses and elapsed time.

    This function automatically logs and prints all loss values
    that are not None, making it method-agnostic (e.g., DICE or others).

    Parameters:
    - opt: options object containing log_path
    - epoch (int): current epoch number
    - elapsed_time (float): time spent for this epoch (in seconds)
    - **loss_dict: named loss tensors (e.g., loss, loss_sum, loss_click, ...)
                   only losses that are not None will be logged
    """

    loss_parts = []

    # Collect all non-None losses
    for loss_name, loss_value in loss_dict.items():
        if loss_value is None:
            continue

        # Convert tensor to scalar
        if hasattr(loss_value, "detach"):
            loss_scalar = loss_value.detach().cpu().numpy()
            if hasattr(loss_scalar, "__len__"):
                loss_scalar = loss_scalar[0]
        else:
            loss_scalar = loss_value

        loss_parts.append(f"{loss_name}: {loss_scalar:.5f}")

    # Join all loss strings
    loss_str = " | ".join(loss_parts)

    # Format epoch time
    time_str = "The time elapse of epoch {:03d} is: {}".format(
        epoch, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    )

    # Write to log file
    with open(opt.log_path, 'a+') as f_log:
        f_log.write(loss_str + '\n')
        f_log.write(time_str + '\n\n')

    # Print to console
    print(loss_str)
    print(time_str)



def import_pybind_module(dataset: str):
    """
    Import the corresponding pybind11 module based on the dataset name.

    Parameters:
    - dataset (str): Name of the dataset. Supported datasets:
      'Douban-movie', 'Amazon-CDs_and_Vinyl', 'Amazon-Music', 'Ciao', 'gowalla'

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
        'gowalla': 'pybind_gowalla'
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset}")

    module_name = dataset_map[dataset]
    return cppimport.imp(module_name)
