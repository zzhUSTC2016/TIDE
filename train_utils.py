import time
import torch
import utils

def train_one_epoch(model, train_loader, optimizer, opt, DICE_alpha=0.1/0.9):
    """
    Train the model for one epoch.

    Returns:
    - loss_sum: total loss for this epoch
    """
    model.train()
    start_time = time.time()
    train_loader.dataset.ng_sample()  # generate training samples
    loss_sum = torch.tensor([0.0]).cuda()

    for batch_id, (user, item_i, item_j, timestamp, split_idx) in enumerate(train_loader, 1):
        user = user.cuda().long()
        item_i = item_i.cuda().long()
        item_j = item_j.cuda().long()
        timestamp = timestamp.cpu().numpy()
        split_idx = split_idx.cpu().numpy()

        optimizer.zero_grad()
        if opt.method == 'IPS':
            loss = model(user, item_i, item_j, timestamp, split_idx)
        elif opt.method == 'DICE':
            loss_click, loss_interest, loss_popularity_1, loss_popularity_2, loss_discrepancy, reg_loss = \
                model(user, item_i, item_j, timestamp, split_idx)
            loss = loss_click + DICE_alpha * (loss_interest + loss_popularity_1 + loss_popularity_2) + 0.01 * loss_discrepancy
            loss += opt.lamb * reg_loss
        else:
            prediction_i, prediction_j, reg_loss = model(user, item_i, item_j, timestamp, split_idx)
            loss = torch.mean(torch.nn.functional.softplus(prediction_j - prediction_i))
            loss += opt.lamb * reg_loss

        loss.backward()
        optimizer.step()
        loss_sum += loss

    elapsed_time = time.time() - start_time
    return loss_sum, elapsed_time


def train_model(opt, train_loader, model, optimizer, evaluator, max_patience=10):
    """
    Full training loop with evaluation, early stopping, and logging.
    """
    best_epoch = evaluator.best_perf['best_epoch']
    DICE_alpha = 0.1 / 0.9

    for epoch in range(opt.epochs):
        DICE_alpha *= 0.9

        loss_sum, elapsed_time = train_one_epoch(model, train_loader, optimizer, opt, DICE_alpha)

        # Logging
        if opt.show_performance:
            utils.log_epoch_loss(opt, epoch, elapsed_time, loss_sum=loss_sum)  # you can add individual losses if needed

        # Evaluation
        model.eval()
        save_flag = evaluator.run_test(epoch)
        if save_flag:
            torch.save(model.state_dict(), opt.model_path)
            if opt.show_performance:
                print(f"Saved model at epoch {epoch}")

        # Early stopping
        if epoch - evaluator.best_perf['best_epoch'] > max_patience:
            print(f"Early stopping at epoch {epoch}")
            with open(opt.log_path, 'a+') as f_log:
                f_log.write(f"Early stopping at epoch {epoch}\n")
            break

    return evaluator.best_perf
