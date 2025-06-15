import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This implementation is based on: https://github.com/manigalati/usad
class EncModule(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        h1 = in_size // 2
        h2 = in_size // 4

        self.linear1 = nn.Linear(in_size, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class DecModule(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        h1 = out_size // 4
        h2 = out_size // 2

        self.linear1 = nn.Linear(latent_size, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class USAD(nn.Module):
    """
      - encoder: in_size → latent_size
      - decoder1: latent_size → in_size
      - decoder2: latent_size → in_size
    """
    def __init__(self, window_size, latent_size):
        super().__init__()
        self.window_size = window_size
        self.latent_size = latent_size

        self.encoder = EncModule(window_size, latent_size)
        self.decoder1 = DecModule(latent_size, window_size)
        self.decoder2 = DecModule(latent_size, window_size)

    def forward(self, x):
        """
        rec1 = decoder1( encoder(x) )
        rec2 = decoder2( encoder( rec1 ) )
        """
        z = self.encoder(x)
        rec1 = self.decoder1(z)
        z_rec1 = self.encoder(rec1)
        rec2 = self.decoder2(z_rec1)
        return rec1, rec2

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        z_w1 = self.encoder(w1)
        w2 = self.decoder2(z_w1)

        alpha = 1.0 / n
        beta = 1.0 - alpha

        loss1 = alpha * torch.mean((batch - w1) ** 2) + beta * torch.mean((batch - w2) ** 2)
        loss2 = alpha * torch.mean((batch - w2) ** 2) - beta * torch.mean((batch - w2) ** 2).detach()
        return loss1, loss2

    def validation_step(self, batch, epoch_index):
        with torch.no_grad():
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            z_w1 = self.encoder(w1)
            w2 = self.decoder2(z_w1)

            alpha = 1.0 / epoch_index
            beta = 1.0 - alpha

            loss1 = alpha * torch.mean((batch - w1) ** 2) + beta * torch.mean((batch - w2) ** 2)
            loss2 = alpha * torch.mean((batch - w2) ** 2) - beta * torch.mean((batch - w2) ** 2).detach()
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss1: {result['val_loss1']:.4f}, val_loss2: {result['val_loss2']:.4f}")

    @staticmethod
    def compute_anomaly_score(x, rec1, rec2, alpha=0.5, beta=0.5):
        """
        alpha * MSE(x, rec1) + beta * MSE(x, rec2).
        """
        with torch.no_grad():
            mse1 = torch.mean((x - rec1) ** 2, dim=1)  # per‐sample
            mse2 = torch.mean((x - rec2) ** 2, dim=1)
            return alpha * mse1 + beta * mse2

    @torch.no_grad()
    def test(self, test_loader, alpha=0.5, beta=0.5):
        all_scores = []
        for batch in test_loader:
            # if loader yields [batch_tensor], unpack
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                batch = batch[0]
            batch = batch.to(device)
            rec1, rec2 = self.forward(batch)
            scores = USAD.compute_anomaly_score(batch, rec1, rec2, alpha, beta)
            all_scores.append(scores.cpu())
        return torch.cat(all_scores, dim=0)
