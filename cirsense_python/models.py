from typing import Tuple


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch is required for deep learning models.") from exc
    return torch, nn


def build_cnn_regressor(input_channels: int = 2):
    torch, nn = require_torch()

    class CNNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.regressor(self.features(x)).squeeze(-1)

    return CNNRegressor()


def build_cnn_lstm_regressor(input_channels: int = 2, hidden_size: int = 64):
    torch, nn = require_torch()

    class CNNLSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, hidden_size, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            # Expected shape: [batch, channel, time]
            encoded = self.encoder(x).transpose(1, 2)
            output, _ = self.lstm(encoded)
            return self.head(output[:, -1, :]).squeeze(-1)

    return CNNLSTMRegressor()


def build_cnn_multitask(input_channels: int, n_classes: int):
    torch, nn = require_torch()

    class CNNMultitask(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.shared = nn.Sequential(nn.Linear(64 * 4 * 4, 96), nn.ReLU(), nn.Dropout(0.1))
            self.regression_head = nn.Linear(96, 1)
            self.classification_head = nn.Linear(96, n_classes)

        def forward(self, x):
            hidden = self.shared(self.features(x))
            return self.regression_head(hidden).squeeze(-1), self.classification_head(hidden)

    return CNNMultitask()


def build_cnn_enhanced_multitask(input_channels: int, n_classes: int):
    torch, nn = require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.05):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            )

        def forward(self, x):
            return self.block(x)

    class CNNEnhancedMultitask(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(input_channels, 32, dropout=0.05),
                ConvBlock(32, 64, dropout=0.08),
                ConvBlock(64, 128, dropout=0.10),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.shared = nn.Sequential(
                nn.Linear(128 * 4 * 4, 192),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(192, 96),
                nn.GELU(),
            )
            self.regression_head = nn.Linear(96, 1)
            self.classification_head = nn.Linear(96, n_classes)

        def forward(self, x):
            hidden = self.shared(self.features(x))
            return self.regression_head(hidden).squeeze(-1), self.classification_head(hidden)

    return CNNEnhancedMultitask()


def build_cnn_lstm_multitask(input_channels: int, n_classes: int, hidden_size: int = 64):
    torch, nn = require_torch()

    class CNNLSTMMultitask(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, hidden_size, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.shared = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.1))
            self.regression_head = nn.Linear(64, 1)
            self.classification_head = nn.Linear(64, n_classes)

        def forward(self, x):
            encoded = self.encoder(x).transpose(1, 2)
            output, _ = self.lstm(encoded)
            hidden = self.shared(output[:, -1, :])
            return self.regression_head(hidden).squeeze(-1), self.classification_head(hidden)

    return CNNLSTMMultitask()


def build_cnn_lstm_enhanced_multitask(input_channels: int, n_classes: int, hidden_size: int = 96):
    torch, nn = require_torch()

    class CNNLSTMEnhancedMultitask(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(input_channels, 48, kernel_size=7, padding=3),
                nn.BatchNorm1d(48),
                nn.GELU(),
                nn.Conv1d(48, hidden_size, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(0.10),
            )
            self.lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.15,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.shared = nn.Sequential(
                nn.Linear(hidden_size * 2, 128),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(128, 64),
                nn.GELU(),
            )
            self.regression_head = nn.Linear(64, 1)
            self.classification_head = nn.Linear(64, n_classes)

        def forward(self, x):
            encoded = self.encoder(x).transpose(1, 2)
            output, _ = self.lstm(encoded)
            attention_logits = self.attention(output)
            attention_weights = torch.softmax(attention_logits, dim=1)
            pooled = torch.sum(output * attention_weights, dim=1)
            hidden = self.shared(pooled)
            return self.regression_head(hidden).squeeze(-1), self.classification_head(hidden)

    return CNNLSTMEnhancedMultitask()


def build_dnn_enhanced_multitask(input_features: int, n_classes: int):
    torch, nn = require_torch()

    class ResidualMLPBlock(nn.Module):
        def __init__(self, width: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.net(x)

    class DNNEnhancedMultitask(nn.Module):
        def __init__(self):
            super().__init__()
            width = 256
            self.input = nn.Sequential(
                nn.LayerNorm(input_features),
                nn.Linear(input_features, width),
                nn.GELU(),
                nn.Dropout(0.10),
            )
            self.blocks = nn.Sequential(
                ResidualMLPBlock(width, 0.15),
                ResidualMLPBlock(width, 0.15),
                ResidualMLPBlock(width, 0.10),
            )
            self.shared = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 128),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.GELU(),
            )
            self.regression_head = nn.Linear(64, 1)
            self.classification_head = nn.Linear(64, n_classes)

        def forward(self, x):
            hidden = self.shared(self.blocks(self.input(x)))
            return self.regression_head(hidden).squeeze(-1), self.classification_head(hidden)

    return DNNEnhancedMultitask()


def build_multiview_fusion_multitask(view_dims: tuple[int, ...], n_classes: int):
    torch, nn = require_torch()

    class ResidualFusionBlock(nn.Module):
        def __init__(self, width: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.net(x)

    class ViewBranch(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class MultiViewFusionMultitask(nn.Module):
        def __init__(self):
            super().__init__()
            self.view_dims = tuple(int(dim) for dim in view_dims)
            branch_widths = []
            for dim in self.view_dims:
                if dim <= 16:
                    branch_widths.append(32)
                elif dim <= 512:
                    branch_widths.append(96)
                else:
                    branch_widths.append(128)
            self.branches = nn.ModuleList(
                ViewBranch(dim, width, dropout=0.12 if dim > 512 else 0.08)
                for dim, width in zip(self.view_dims, branch_widths)
            )
            fused_width = sum(branch_widths)
            hidden_width = 192
            self.fusion = nn.Sequential(
                nn.LayerNorm(fused_width),
                nn.Linear(fused_width, hidden_width),
                nn.GELU(),
                nn.Dropout(0.15),
                ResidualFusionBlock(hidden_width, 0.12),
                ResidualFusionBlock(hidden_width, 0.10),
                nn.LayerNorm(hidden_width),
                nn.Linear(hidden_width, 96),
                nn.GELU(),
            )
            self.regression_head = nn.Linear(96, 1)
            self.classification_head = nn.Linear(96, n_classes)

        def forward(self, x):
            chunks = torch.split(x, self.view_dims, dim=1)
            encoded = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
            hidden = self.fusion(torch.cat(encoded, dim=1))
            return self.regression_head(hidden).squeeze(-1), self.classification_head(hidden)

    return MultiViewFusionMultitask()


def model_input_shape_hint(model_name: str) -> Tuple[str, str]:
    if model_name == "cnn":
        return "CNNRegressor", "[batch, channel, time, tap]"
    if model_name == "cnn_lstm":
        return "CNNLSTMRegressor", "[batch, channel, time]"
    raise ValueError(f"Unknown model name: {model_name}")
