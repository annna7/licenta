import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM_AR(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float = 0.0,
    ):
        """
        input_size: number of features per time step (e.g., number of sensors)
        hidden_size: LSTM hidden dimension
        num_layers: number of LSTM layers
        dropout: dropout between LSTM layers (0.0 means no dropout)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden=None):
        out_seq, (h_n, c_n) = self.lstm(x, hidden)
        last_hidden = out_seq[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.fc(last_hidden)  # (batch_size, input_size)
        return y_pred, (h_n, c_n)

    def training_step(self, batch_window, next_step):
        batch_window = batch_window.to(device)
        next_step = next_step.to(device)
        y_pred, _ = self.forward(batch_window)
        loss = nn.functional.mse_loss(y_pred, next_step)
        return loss

    @torch.no_grad()
    def predict(self, batch_window):
        batch_window = batch_window.to(device)
        y_pred, _ = self.forward(batch_window)
        return y_pred.cpu()

    @torch.no_grad()
    def test(self, test_loader):
        all_scores = []

        for batch in test_loader:
            batch_window, next_step = batch
            y_pred = self.predict(batch_window)
            next_step = next_step.to(torch.float32)
            mse_per_sample = torch.mean((y_pred - next_step.cpu()) ** 2, dim=1)
            all_scores.append(mse_per_sample)
        return torch.cat(all_scores, dim=0)
