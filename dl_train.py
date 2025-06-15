import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_test import test_gdn, test_usad, test_lstm_ar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def loss_func(y_pred, y_true):
    return F.mse_loss(y_pred, y_true, reduction="mean")


def train_gdn(
        model,
        save_path: str,
        config: dict,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        feature_map,
        test_dataloader: DataLoader,
        test_dataset,
        train_dataset,
        dataset_name: str,
):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("decay", 0.0),
    )
    best_val_f1 = 0.0
    min_loss = 1e8

    for epoch in tqdm(range(config["epoch"]), desc="GDN epochs"):
        model.train()
        acc_loss = 0.0

        for x, labels, attack_labels, edge_index in train_dataloader:
            x = x.float().to(device)
            labels = labels.float().to(device)
            edge_index = edge_index.to(device)

            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()

        avg_train_loss = acc_loss / len(train_dataloader)
        print(f"epoch ({epoch + 1}/{config['epoch']}) train_loss: {avg_train_loss:.8f}")

        if val_dataloader is not None:
            model.eval()
            val_loss, _ = test_gdn(model, val_dataloader)
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
        else:
            if acc_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acc_loss

    return {"best_val_f1": best_val_f1}


def train_usad(
        model: nn.Module,
        save_path: str,
        config: dict,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    optimizer1 = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=config.get("lr", 1e-3),
    )
    optimizer2 = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=config.get("lr", 1e-3),
    )

    n_epochs = config["epoch"]
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, n_epochs + 1), desc="USAD epochs"):
        model.train()
        acc_loss = 0.0
        batch_count = 0

        for x_flat, _dummy_label in train_dataloader:
            x_flat = x_flat.float().to(device)

            loss1, loss2 = model.training_step(x_flat, epoch)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss1, loss2 = model.training_step(x_flat, epoch)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            acc_loss += (loss1.item() + loss2.item())
            batch_count += 1

        avg_train_loss = acc_loss / batch_count
        print(f"epoch ({epoch}/{n_epochs}) train_loss: {avg_train_loss:.8f}")

        if val_dataloader is not None:
            model.eval()
            val_loss, _ = test_usad(model, val_dataloader)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), save_path)
                best_val_loss = val_loss
        else:
            if acc_loss < best_val_loss:
                torch.save(model.state_dict(), save_path)
                best_val_loss = acc_loss

    return {"best_val_loss": best_val_loss}


def train_lstm_ar(
        model: nn.Module,
        save_path: str,
        config: dict,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    optimizer = optim.Adam(
        model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("decay", 0.0)
    )

    n_epochs = config["epoch"]
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, n_epochs + 1), desc="LSTM_AR epochs"):
        model.train()
        acc_loss = 0.0
        batch_count = 0

        for window, next_step, dummy_train in train_dataloader:
            window = window.float().to(device)
            next_step = next_step.float().to(device)

            optimizer.zero_grad()
            loss = model.training_step(window, next_step)
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()
            batch_count += 1

        avg_train_loss = acc_loss / batch_count
        print(f"epoch ({epoch}/{n_epochs}) train_loss: {avg_train_loss:.8f}")

        if val_dataloader is not None:
            model.eval()
            val_loss, _ = test_lstm_ar(model, val_dataloader)
            print(f"    Validation → epoch ({epoch}/{n_epochs}) val_loss: {val_loss:.8f}")
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), save_path)
                best_val_loss = val_loss
        else:
            if acc_loss < best_val_loss:
                torch.save(model.state_dict(), save_path)
                best_val_loss = acc_loss

    return {"best_val_loss": best_val_loss}
