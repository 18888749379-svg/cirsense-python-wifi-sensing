from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import TrainConfig
from models import require_torch


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int = 0
    best_val_loss: float = float("nan")


def train_regressor(
    model,
    x: np.ndarray,
    y: np.ndarray,
    train_config: TrainConfig,
    device: Optional[str] = None,
) -> TrainHistory:
    torch, nn = require_torch()
    from torch.utils.data import DataLoader, TensorDataset, random_split

    torch.manual_seed(train_config.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)

    val_size = int(len(dataset) * train_config.validation_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_config.batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = nn.MSELoss()
    history = TrainHistory(train_loss=[], val_loss=[])

    for _ in range(train_config.epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = criterion(model(batch_x), batch_y)
                val_losses.append(float(loss.detach().cpu()))

        history.train_loss.append(float(np.mean(train_losses)))
        history.val_loss.append(float(np.mean(val_losses)) if val_losses else float("nan"))

    return history


def train_multitask_model(
    model,
    x: np.ndarray,
    y_regression: np.ndarray,
    y_class: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_config: TrainConfig,
    classification_weight: float = 0.25,
    class_loss_weights: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> TrainHistory:
    torch, nn = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(train_config.random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_indices = np.asarray(train_indices, dtype=int)
    val_indices = np.asarray(val_indices, dtype=int)
    if len(train_indices) == 0:
        raise ValueError("Multitask training requires at least one training sample.")

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_reg_tensor = torch.tensor(y_regression, dtype=torch.float32)
    y_class_tensor = torch.tensor(y_class, dtype=torch.long)
    train_set = TensorDataset(x_tensor[train_indices], y_reg_tensor[train_indices], y_class_tensor[train_indices])
    val_set = TensorDataset(x_tensor[val_indices], y_reg_tensor[val_indices], y_class_tensor[val_indices])
    train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_config.batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=1e-4)
    regression_loss = nn.SmoothL1Loss()
    if class_loss_weights is not None:
        weight_tensor = torch.tensor(class_loss_weights, dtype=torch.float32, device=device)
        classification_loss = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        classification_loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )
    history = TrainHistory(train_loss=[], val_loss=[])
    best_score = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for epoch in range(train_config.epochs):
        model.train()
        train_losses = []
        for batch_x, batch_reg, batch_class in train_loader:
            batch_x = batch_x.to(device)
            batch_reg = batch_reg.to(device)
            batch_class = batch_class.to(device)
            optimizer.zero_grad()
            pred_reg, logits = model(batch_x)
            loss = regression_loss(pred_reg, batch_reg)
            loss = loss + classification_weight * classification_loss(logits, batch_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_reg, batch_class in val_loader:
                batch_x = batch_x.to(device)
                batch_reg = batch_reg.to(device)
                batch_class = batch_class.to(device)
                pred_reg, logits = model(batch_x)
                loss = regression_loss(pred_reg, batch_reg)
                loss = loss + classification_weight * classification_loss(logits, batch_class)
                val_losses.append(float(loss.detach().cpu()))

        train_mean = float(np.mean(train_losses))
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        score = val_mean if np.isfinite(val_mean) else train_mean
        scheduler.step(score)
        if score < best_score:
            best_score = score
            history.best_epoch = epoch + 1
            history.best_val_loss = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        history.train_loss.append(train_mean)
        history.val_loss.append(val_mean)

    model.load_state_dict(best_state)
    return history


def predict_multitask_model(model, x: np.ndarray, device: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    torch, _ = require_torch()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred_reg, logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        pred_class = torch.argmax(logits, dim=1)
    return pred_reg.detach().cpu().numpy(), pred_class.detach().cpu().numpy()
