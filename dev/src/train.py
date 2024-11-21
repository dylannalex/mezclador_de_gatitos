import os
import json
from typing import Union

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def update_history_csv(history_path: str, history_dict: dict):
    """
    Update the CSV file with training history.

    If the history file does not exist, it creates a new one. If it exists,
    it merges the new history with the existing history based on epochs.

    Args:
        history_path (str): The path to the history CSV file.
        history_dict (dict): A dictionary containing epoch history data.
    """
    if not os.path.exists(history_path):
        history = pd.DataFrame(history_dict)
        history.to_csv(history_path, index=False)
        return

    history = pd.read_csv(history_path)
    new_history = pd.DataFrame(history_dict)
    merged_history = pd.merge(
        new_history, history, on="epoch", how="outer", suffixes=("_x", "_y")
    )
    merged_history["train_loss"] = merged_history["train_loss_x"].combine_first(
        merged_history["train_loss_y"]
    )

    if history_dict["val_loss"][-1] is not None:
        merged_history["val_loss"] = merged_history["val_loss_x"].combine_first(
            merged_history["val_loss_y"]
        )
        merged_history = merged_history[["epoch", "train_loss", "val_loss"]]

    else:
        merged_history = merged_history[["epoch", "train_loss"]]

    merged_history.to_csv(history_path, index=False)


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer,
    device: str,
    loss_function: nn.Module,
):
    """
    Train the nn.Module for one epoch.

    Args:
        model (nn.Module): The Variational Autoencoder model.
        data_loader (DataLoader): DataLoader for the training data.
        optimizer: The optimizer used for training.
        device (str): The device to perform computations on (e.g., 'cpu' or 'cuda').
        loss_function (Callable): The loss function used for training.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for x, _ in tqdm(data_loader):
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        x_hat, mean, log_var = model(x)

        # Compute loss and backpropagate
        loss_value = loss_function(x, x_hat, mean, log_var)
        loss_value.backward()
        optimizer.step()

        # Accumulate loss for epoch
        total_loss += loss_value.item() * x.size(0)
        num_samples += x.size(0)

    avg_loss = total_loss / num_samples
    return avg_loss


def val_epoch(
    model: nn.Module, data_loader: DataLoader, device: str, loss_function: nn.Module
):
    """
    Validate the nn.Module for one epoch.

    Args:
        model (nn.Module): The Variational Autoencoder model.
        data_loader (DataLoader): DataLoader for the validation data.
        device (str): The device to perform computations on (e.g., 'cpu' or 'cuda').
        loss_function (Callable): The loss function used for validation.

    Returns:
        float: The average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for x, _ in tqdm(data_loader):
            x = x.to(device)

            # Forward pass
            x_hat, mean, log_var = model(x)

            # Compute validation loss
            loss_value = loss_function(x, x_hat, mean, log_var)

            # Accumulate loss for validation
            total_loss += loss_value.item() * x.size(0)
            num_samples += x.size(0)

    avg_loss = total_loss / num_samples
    return avg_loss


def build_model_dir(weights_dir: str, epochs_weights_dir: str):
    """
    Create directories for saving model weights.

    Args:
        weights_dir (str): The base directory for weights.
        epochs_weights_dir (str): The directory for epoch-specific weights.
    """
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    if not os.path.exists(epochs_weights_dir):
        os.makedirs(epochs_weights_dir)


def get_weights_epoch_num(weights_path: str):
    """
    Extract the epoch number from the weights file name.

    Args:
        weights_path (str): The path to the weights file.

    Returns:
        int: The epoch number extracted from the file name.
    """
    return int(os.path.basename(weights_path).split("_")[0])


def find_latest_weights(weights_dir: str):
    """
    Find the latest weights file in the given directory.

    Args:
        weights_dir (str): The directory to search for weights files.

    Returns:
        str or None: The path to the latest weights file or None if no files are found.
    """
    saved_weights = [
        os.path.join(weights_dir, f)
        for f in os.listdir(weights_dir)
        if f.endswith(".pt")
    ]

    latest_weights = (
        max(saved_weights, key=get_weights_epoch_num) if saved_weights else None
    )
    return latest_weights


def get_best_models_stats(model_dir: str):
    """
    Find the best train and validation loss from the best-models.json file.

    Args:
        model_dir (str): The directory containing the best-models.json file.

    Returns:
        tuple: A tuple containing the best train loss and best validation loss.
    """
    best_models_path = os.path.join(model_dir, "best_models_stats.json")

    if not os.path.exists(best_models_path):
        return {
            "train": {
                "train_loss": float("inf"),
                "val_loss": float("inf"),
                "epoch": 0,
            },
            "val": {
                "train_loss": float("inf"),
                "val_loss": float("inf"),
                "epoch": 0,
            },
        }

    with open(best_models_path, "r") as f:
        best_models_stats = json.load(f)

    return best_models_stats


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Union[DataLoader, None],
    epochs: int,
    epochs_per_checkpoint: int,
    device: str,
    model_dir: str,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
):
    """
    Train the nn.Module model for a specified number of epochs.

    Args:
        model (nn.Module): The Variational Autoencoder model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        epochs (int): The total number of epochs for training.
        epochs_per_checkpoint (int): How often to save model checkpoints.
        device (str): The device to perform computations on (e.g., 'cpu' or 'cuda').
        model_dir (str): Directory to save model weights and history.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_function (nn.Module): The loss function used for training.

    Returns:
        dict: A dictionary containing training history (epoch, train_loss, val_loss).
    """
    # Create directories for saving weights
    epochs_weights_dir = os.path.join(model_dir, "epochs")
    history_csv_path = os.path.join(model_dir, "history.csv")
    build_model_dir(model_dir, epochs_weights_dir)

    # Load latest epoch weights if available
    latest_weights = find_latest_weights(epochs_weights_dir)
    if latest_weights:
        print(
            f"[-] Resuming training from epoch: {get_weights_epoch_num(latest_weights)}"
        )
        model.load_state_dict(
            torch.load(latest_weights, map_location=torch.device(device))
        )
        epochs_range = range(get_weights_epoch_num(latest_weights) + 1, epochs + 1)
    else:
        epochs_range = range(1, epochs + 1)

    # Load best model stats
    best_models_stats = get_best_models_stats(model_dir)

    # Train the model
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    for epoch in epochs_range:
        print(f"[-] Starting epoch: {epoch}")
        history["epoch"].append(epoch)

        print(f"[-] Training:", end=" ")
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_function)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            print(f"[-] Validating:", end=" ")
            val_loss = val_epoch(model, val_loader, device, loss_function)
            history["val_loss"].append(val_loss)
        else:
            val_loss = None
            history["val_loss"].append(val_loss)

        # Update history CSV
        update_history_csv(history_csv_path, history)

        # Save the model after each epoch
        if epoch % epochs_per_checkpoint == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    epochs_weights_dir, f"{epoch}_train-{train_loss}_val-{val_loss}.pt"
                ),
            )

        # Save the best model based on validation loss
        if val_loader is not None:
            if val_loss < best_models_stats["val"]["val_loss"]:
                print(f"[-] Saving best model based on validation loss: {val_loss:.6f}")

                best_models_stats["val"]["train_loss"] = train_loss
                best_models_stats["val"]["val_loss"] = val_loss
                best_models_stats["val"]["epoch"] = epoch

                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, f"best-val.pt"),
                )

                # Save best models stats
                with open(os.path.join(model_dir, "best_models_stats.json"), "w") as f:
                    json.dump(best_models_stats, f)

        # Save the best model based on train loss
        if train_loss < best_models_stats["train"]["train_loss"]:
            print(f"[-] Saving best model based on train loss: {train_loss:.6f}")

            best_models_stats["train"]["train_loss"] = train_loss
            best_models_stats["train"]["val_loss"] = val_loss
            best_models_stats["train"]["epoch"] = epoch

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f"best-train.pt"),
            )

            # Save best models stats
            with open(os.path.join(model_dir, "best_models_stats.json"), "w") as f:
                json.dump(best_models_stats, f)

        if val_loss is None:
            print(f"Train Loss: {train_loss:.6f}\n\n")
        else:
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n\n")

    return history
