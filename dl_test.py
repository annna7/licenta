import time

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_gdn(model, dataloader):
    loss_func = nn.MSELoss(reduction="mean")

    test_loss_list = []

    t_test_predicted = None
    t_test_ground = None
    t_test_labels = None

    test_len = len(dataloader)
    model.eval()

    i = 0
    for x, y, labels, edge_index in dataloader:
        x = x.to(device).float()
        y = y.to(device).float()
        labels = labels.to(device).float()
        edge_index = edge_index.to(device)

        with torch.no_grad():
            predicted = model(x, edge_index).to(device)
            loss = loss_func(predicted, y)

            labels_exp = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if t_test_predicted is None:
                t_test_predicted = predicted.detach().cpu()
                t_test_ground = y.detach().cpu()
                t_test_labels = labels_exp.detach().cpu()
            else:
                t_test_predicted = torch.cat((t_test_predicted, predicted.detach().cpu()), dim=0)
                t_test_ground = torch.cat((t_test_ground, y.detach().cpu()), dim=0)
                t_test_labels = torch.cat((t_test_labels, labels_exp.detach().cpu()), dim=0)

        test_loss_list.append(loss.item())
        i += 1

    test_predicted_list = t_test_predicted.tolist()
    test_ground_list = t_test_ground.tolist()
    test_labels_list = t_test_labels.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)
    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]


def test_usad(model, dataloader, alpha=0.5, beta=0.5):
    loss_func = nn.MSELoss(reduction="none")

    test_loss_list = []
    now = time.time()

    # Accumulators
    t_pred = None
    t_ground = None
    t_labels = None

    i = 0
    model.eval()

    for x_flat, labels in dataloader:
        x_flat = x_flat.to(device).float()
        labels = labels.to(device).float()

        with torch.no_grad():
            rec1, rec2 = model(x_flat)  # both (batch_size, window_len)
            mse1 = torch.mean((rec1 - x_flat) ** 2, dim=1)  # (batch_size,)
            batch_loss = torch.mean(mse1).item()

            rec1_cpu = rec1.cpu()
            x_flat_cpu = x_flat.cpu()

            labels_exp = labels.unsqueeze(1).repeat(1, rec1_cpu.shape[1]).cpu()

            if t_pred is None:
                t_pred = rec1_cpu.clone()
                t_ground = x_flat_cpu.clone()
                t_labels = labels_exp.clone() if labels_exp is not None else None
            else:
                t_pred = torch.cat((t_pred, rec1_cpu), dim=0)
                t_ground = torch.cat((t_ground, x_flat_cpu), dim=0)
                if labels_exp is not None:
                    t_labels = torch.cat((t_labels, labels_exp), dim=0)

        test_loss_list.append(batch_loss)
        i += 1

    predicted_list = t_pred.tolist()
    ground_list = t_ground.tolist()
    labels_list = t_labels.tolist() if t_labels is not None else []

    avg_loss = sum(test_loss_list) / len(test_loss_list)
    return avg_loss, [predicted_list, ground_list, labels_list]


def test_lstm_ar(model, dataloader):
    loss_fn = nn.MSELoss(reduction="none")
    t_pred, t_ground, t_labels = None, None, None
    total_loss = []

    model.eval()
    for window, next_step, labels in dataloader:
        window = window.to(device)
        next_step = next_step.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            y_pred = model.predict(window).to(device)
            # per-sample MSE
            mse_el = loss_fn(y_pred, next_step)  # (B, F)
            mse_samp = mse_el.mean(dim=1)  # (B,)
            total_loss.append(mse_samp.mean().item())

            # accumulate
            if t_pred is None:
                t_pred = y_pred.cpu()
                t_ground = next_step.cpu()
                t_labels = labels.unsqueeze(1).repeat(1, y_pred.size(1)).cpu()
            else:
                t_pred = torch.cat((t_pred, y_pred.cpu()), dim=0)
                t_ground = torch.cat((t_ground, next_step.cpu()), dim=0)
                t_labels = torch.cat((t_labels, labels.unsqueeze(1)
                                      .repeat(1, y_pred.size(1))
                                      .cpu()), dim=0)

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss, [t_pred.tolist(), t_ground.tolist(), t_labels.tolist()]
